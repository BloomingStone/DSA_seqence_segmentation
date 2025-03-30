import os
from pathlib import Path
from time import time
import tomllib

import torch
from monai.losses import DiceCELoss
from monai.metrics import CumulativeAverage
from monai.data import DataLoader, ThreadDataLoader

# todo 不要用这种方法导入params, 使用显式load
from .dataloader import DataLoaderManager
from .model import get_model_inited_by_base_model_checkpoints
from .single_epochs import train_single_epoch, validate_single_epoch, test_single_epoch
from ...core.utils.log import log_info, log_expected_time
from ...core.utils.task_type import TaskType
from ...core.utils.log import init_logger
from ...core.utils.tensorboard import init_writer

_task_dir = Path(__file__).parent

init_logger(_task_dir)
init_writer(_task_dir)


def main():
    task_dir = Path(__file__).parent
    with open(task_dir / "params.toml", "rb") as f:
        params = tomllib.load(f)

    checkpoint_dir = task_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = params["device"]["CUDA_VISIBLE_DEVICES"]
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    model = get_model_inited_by_base_model_checkpoints(
        Path(params["model"]['basemodel_checkpoint_path']),
        num_classes=1
    ).to('cuda')
    loss_function = DiceCELoss(sigmoid=True)

    dataloader_manager = DataLoaderManager(
        Path(params["dataset"]["image_dir"]),
        Path(params["dataset"]["label_dir"]),
        Path(params["dataset"]["data_info_csv_path"]),
        params["dataset"]["n_splits"],
        params["dataset"]["split_random_seed"]
    )

    def get_dataloader(task_type: TaskType) -> ThreadDataLoader:
        return dataloader_manager.get_dataloader(
            task_type,
            params["dataset"]["fold"],
            params[task_type]["batch_size"],
            params[task_type]["shuffle"]
        )

    train_dataloader = get_dataloader(TaskType.Train)
    valid_dataloader = get_dataloader(TaskType.Valid)
    test_dataloader = get_dataloader(TaskType.Test)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=3e-5
    )
    scaler = torch.amp.GradScaler('cuda')

    n_epochs = params["train"]["n_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda x: (1 - x / n_epochs) ** 0.9
    )

    dice_best = 0
    dice_best_epoch = 0
    best_checkpoint_path: None | Path = None
    time_avg = CumulativeAverage()
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

        log_info(f'validation begin')
        _, dice = validate_single_epoch(
            model,
            loss_function,
            valid_dataloader,
            epoch,
            n_epochs
        )

        time_avg.append(time() - time_start)

        if dice > dice_best:
            dice_best = dice
            dice_best_epoch = epoch
            best_checkpoint_path = checkpoint_dir / 'best.pth'
            torch.save(
                model.state_dict(),
                best_checkpoint_path
            )
        log_info(f'up to now the best dice is: {dice_best:.4f} at epoch: {dice_best_epoch+1}')

        if epoch % params["train"]["saving_interval"] == 0:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f'latest.pth'
            )

        log_expected_time(
            time_avg.aggregate().item(),
            epoch,
            n_epochs
        )

    log_info(f'training finished, best dice: {dice_best:.4f}')
    model.load_state_dict(torch.load(best_checkpoint_path, map_location='cuda', weights_only=True))
    test_single_epoch(
        model,
        loss_function,
        test_dataloader,
        max_saved_image=params["test"]["max_saved_image"]
    )

if __name__ == '__main__':
    main()