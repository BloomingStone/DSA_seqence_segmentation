import os
from pathlib import Path
from time import time
import tomllib

import torch
from monai.losses import DiceCELoss     # type: ignore
from monai.metrics import CumulativeAverage # type: ignore
from monai.data import DataLoader # type: ignore

from .dataloader import DataLoaderManager
from .single_epochs import train_single_epoch, validate_single_epoch, test_single_epoch
from ...core.base_models.ResEncUNet_pure_torch import ResEncUNet
from ...core.utils.log import log_info, log_expected_time
from ...core.utils.task_type import TaskType
from ...core.utils.log import init_logger
from ...core.utils.tensorboard import init_writer


def main():
    task_dir = Path(__file__).parent
    with open(task_dir / "params.toml", "rb") as f:
        params = tomllib.load(f)

    checkpoint_dir = task_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = params["device"]["CUDA_VISIBLE_DEVICES"]
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    model = ResEncUNet(output_channels=1).to('cuda')
    loss_function = DiceCELoss(include_background=False, sigmoid=True)

    dataloader_manager = DataLoaderManager(
        train_image_dir=Path(params["dataset"]["train_images_dir"]),
        train_label_dir=Path(params["dataset"]["train_labels_dir"]),
        test_image_dir=Path(params["dataset"]["test_images_dir"]),
        test_label_dir=Path(params["dataset"]["test_labels_dir"]),
    )

    def get_dataloader(task_type: TaskType) -> DataLoader:
        return dataloader_manager.get_dataloader(
            task_type,
            params[task_type]["batch_size"],
            params[task_type]["shuffle"]
        )

    train_dataloader = get_dataloader(TaskType.Train)
    valid_dataloader = get_dataloader(TaskType.Valid)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler('cuda')   # type: ignore

    n_epochs = params["train"]["n_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda x: (1 - x / n_epochs) ** 0.9
    )

    dice_best = 0
    dice_best_epoch = 0
    best_checkpoint_path: None | Path = None
    train_time_avg = CumulativeAverage()
    valid_time_avg = CumulativeAverage()
    for epoch in range(n_epochs):
        time_start = time()
        log_info(f'Training {epoch + 1}/{n_epochs}')
        train_single_epoch(
            model,
            loss_function,
            train_dataloader,
            optimizer,
            scaler,
            epoch,
            n_epochs
        )
        scheduler.step()
        train_time_avg.append(time() - time_start)

        if epoch % params["train"]["saving_interval"] == 0:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f'latest.pth'
            )

        if epoch % params["valid"]["valid_interval"] == 0:
            log_info(f'validation begin')
            time_start = time()
            _, dice = validate_single_epoch(
                model,
                loss_function,
                valid_dataloader,
                epoch,
                n_epochs
            )
            valid_time_avg.append(time() - time_start)

            log_expected_time(
                train_time_avg.aggregate().item(),
                valid_time_avg.aggregate().item(),
                epoch,
                n_epochs,
                params["valid"]["valid_interval"]
            )

            if dice > dice_best:
                dice_best = dice
                dice_best_epoch = epoch
                best_checkpoint_path = checkpoint_dir / 'best.pth'
                torch.save(
                    model.state_dict(),
                    best_checkpoint_path
                )
            log_info(f'up to now the best dice is: {dice_best:.4f} at epoch: {dice_best_epoch+1}')

    log_info(f'training finished, best dice: {dice_best:.4f}')
    assert best_checkpoint_path is not None
    test_dataloader = get_dataloader(TaskType.Valid)
    model.load_state_dict(torch.load(best_checkpoint_path, map_location='cuda', weights_only=True))
    test_single_epoch(
        model,
        loss_function,
        test_dataloader,
        max_saved_image=params["test"]["max_saved_image"]
    )

if __name__ == '__main__':
    _task_dir = Path(__file__).parent
    init_logger(_task_dir)
    init_writer(_task_dir)
    main()