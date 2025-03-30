from time import time
from pathlib import Path

import numpy as np
from monai.data import ThreadDataLoader
from monai.transforms import AsDiscrete
from torch import nn
import torch
from monai.metrics.cumulative_average import CumulativeAverage
from monai.metrics import DiceHelper

from .image_type import ImageType
from ...core.utils.log import log_single_batch_info, log_valid_end_info, log_info
from ...core.utils.tensorboard import record_scalar, record_learning_rate, record_image_label_prediction
from ...core.utils.task_type import TaskType
from ...core.utils.metrics import cal_clDice_temp, cal_hausdorff_temp, cal_continuity


def train_single_epoch(
        model: nn.Module,
        loss_function: nn.Module,
        data_loader: ThreadDataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp,
        epoch: int,
        n_epochs: int,
) -> None:
    model.train()
    time_start = time()
    loss_avg = CumulativeAverage()
    dice_avg = CumulativeAverage()
    dice_helper = DiceHelper(sigmoid=True, activate=True, get_not_nans=False)

    for batch_id, batch_data in enumerate(data_loader):
        x = batch_data[ImageType.Image].as_tensor().to('cuda')
        y = batch_data[ImageType.Label].as_tensor().to('cuda')
        first_frame = batch_data[ImageType.FirstImage].as_tensor().to('cuda')
        with torch.amp.autocast('cuda'):
            logit = model(x, first_frame)
            loss = torch.stack([
                0.5 ** i * loss_function(o, y) for i, o in enumerate(logit)
            ])
            loss = loss.sum()

            dice = dice_helper(logit[0], y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_avg.append(loss.item())
        loss_avg.append(dice.item())

        time_end = time()
        if batch_id % 10 == 0:
            log_single_batch_info(
                epoch,
                n_epochs,
                batch_id,
                len(data_loader),
                loss.item(),
                dice.item(),
                time_end - time_start,
            )
        time_start = time_end

    loss_avg = loss_avg.aggregate().item()
    dice_avg = dice_avg.aggregate()
    record_scalar(TaskType.Train, "loss", loss_avg, epoch)
    record_scalar(TaskType.Train, "dice", dice_avg, epoch)
    record_learning_rate(optimizer, epoch)

def validate_single_epoch(
        model: nn.Module,
        loss_function: nn.Module,
        data_loader: ThreadDataLoader,
        epoch: int,
        n_epochs: int,
) -> tuple[float, float]:
    model.eval()
    time_start = time()
    loss_avg = CumulativeAverage()
    dice_avg = CumulativeAverage()
    dice_helper = DiceHelper(sigmoid=True, activate=True, get_not_nans=False)
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            x = batch_data[ImageType.Image].as_tensor().to('cuda')
            y = batch_data[ImageType.Label].as_tensor().to('cuda')
            meta = batch_data[ImageType.Image].meta
            first_frame = batch_data[ImageType.FirstImage].as_tensor().to('cuda')
            with torch.amp.autocast('cuda'):
                logit = model(x, first_frame)
                loss = loss_function(logit, y)
                dice = dice_helper(logit, y)

            time_end = time()
            log_single_batch_info(
                epoch,
                n_epochs,
                batch_id,
                len(data_loader),
                loss.item(),
                dice.item(),
                time_end - time_start,
                f"{meta['image_name']} - {meta['slice']}"
            )
            time_start = time_end
            loss_avg.append(loss.item())
            dice_avg.append(dice.item())

    loss_avg_item = loss_avg.aggregate().item()
    dice_avg_item = dice_avg.aggregate().item()
    log_valid_end_info(epoch + 1, loss_avg_item, dice_avg_item)
    record_scalar(TaskType.Valid, "loss", loss_avg_item, epoch)
    record_scalar(TaskType.Valid, "dice", dice_avg_item, epoch)
    return loss_avg_item, dice_avg_item


def test_single_epoch(
        model: nn.Module,
        loss_function: nn.Module,
        data_loader: ThreadDataLoader,
        max_saved_image: int = 20,
        saved_iamge_to_tensorboard: bool = False,
        image_output_dir: Path = None
) -> tuple[float, float]:
    model.eval()
    # todo 将这些东西移到metrix模块中
    loss_avg = CumulativeAverage()
    dice_avg = CumulativeAverage()
    cldice_avg = CumulativeAverage()
    hd95_avg = CumulativeAverage()
    continuity_avg = CumulativeAverage()
    time_avg = CumulativeAverage()
    dice_helper = DiceHelper(sigmoid=True, activate=True, get_not_nans=False)
    as_discrete = AsDiscrete(threshold=0.5)

    time_start = time()
    image_count = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            x = batch_data[ImageType.Image].as_tensor().to('cuda')
            y = batch_data[ImageType.Label].as_tensor().to('cuda')
            meta = batch_data[ImageType.Image].meta
            first_frame = batch_data[ImageType.FirstImage].as_tensor().to('cuda')

            logit_map = model(x, first_frame)
            t = time() - time_start
            time_start = time()

            assert isinstance(logit_map, torch.Tensor)
            loss = loss_function(logit_map, y)
            prediction = as_discrete(logit_map)
            dice = dice_helper(logit_map, y)
            cldice = cal_clDice_temp(prediction, y)
            hd95 = cal_hausdorff_temp(prediction, y)
            continuity = cal_continuity(prediction, y)

            loss_avg.append(loss)
            dice_avg.append(dice)
            cldice_avg.append(cldice)
            hd95_avg.append(hd95)
            continuity_avg.append(continuity)
            time_avg.append(t)


            info = {
                "iter": f"{batch_id+1}/{len(data_loader)}",
                "loss": loss.item(),
                "dice": dice.item(),
                "cldice": cldice.item(),
                "hd95": hd95.item(),
                "continuity": continuity,
                "time": t,
                "image": f"{meta['image_name'][0]} - {meta['slice'].item()}"
            }

            log_info('\t'.join([f'{k}: {v:.6f}' if isinstance(v, float) else f'{k}: {v}' for k, v in info.items()]))

            if (
                (saved_iamge_to_tensorboard or image_output_dir is not None) 
                and image_count < max_saved_image 
                and (continuity <= 0 or dice < 0.85)
            ):
                image_name = batch_data[ImageType.Image].meta["image_name"][0]
                slice = batch_data[ImageType.Image].meta["slice"].item()
                if saved_iamge_to_tensorboard:
                    record_image_label_prediction(
                        f'{image_name} - {slice}',
                        x.squeeze().cpu().numpy(),
                        y.squeeze().cpu().numpy(),
                        prediction.squeeze().cpu().numpy(),
                    )
                # todo 在 core 中新增一个专门用来记录图片的类或者方法，支持导出gif, png, nii.gz等选择
                if image_output_dir is not None:
                    from PIL import Image
                    image_output_dir.mkdir(exist_ok=True, parents=True)
                    image = x.squeeze().cpu().numpy()
                    image = (image - image.min()) / (image.max() - image.min()) * 255
                    image = image.astype(np.uint8)
                    predict = prediction.squeeze().cpu().numpy().astype(np.uint8)
                    predict[predict== 1] = 255
                    label = y.squeeze().cpu().numpy().astype(np.uint8)
                    label[label== 1] = 255
                    image_output_path = image_output_dir / f'{image_name}_{slice}_image.png'
                    predict_output_path = image_output_dir / f'{image_name}_{slice}_predict.png'
                    label_output_path = image_output_dir / f'{image_name}_{slice}_label.png'
                    image = Image.fromarray(image)
                    predict = Image.fromarray(predict)
                    label = Image.fromarray(label)
                    image.save(image_output_path)
                    predict.save(predict_output_path)
                    label.save(label_output_path)


    info = {
        "loss": loss_avg.aggregate(),
        "dice": dice_avg.aggregate().item(),
        "cldice": cldice_avg.aggregate().item(),
        "hd95": hd95_avg.aggregate().item(),
        "continuity": continuity_avg.aggregate().item(),
        "time": time_avg.aggregate().item(),
    }
    log_info('Done: Average indicators are:')
    log_info('\t'.join([f'{k}: {v:.6f}' for k, v in info.items()]))

    return loss_avg.aggregate().item(), dice_avg.aggregate().item()