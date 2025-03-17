import os
from pathlib import Path
from time import time

import torch
from monai.losses import DiceCELoss
from monai.metrics import CumulativeAverage

from . import params
from .dataloader import get_dataloaders
from .model import get_model_inited_by_base_model_checkpoints
from .single_epochs import train_single_epoch, validate_single_epoch, test_single_epoch
from ...core.utils.log import log_info, log_expected_time
from ...core.utils.task_type import TaskType


def main():
    task_dir = Path(__file__).parent
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
    dataloaders = get_dataloaders(params['dataset']['fold'])

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
    time_avg = CumulativeAverage()
    for epoch in range(n_epochs):
        time_start = time()
        log_info(f'Training epoch {epoch + 1}')
        train_single_epoch(
            model,
            loss_function,
            dataloaders[TaskType.Train],
            optimizer,
            scaler,
            epoch
        )
        scheduler.step()

        log_info(f'validation begin')
        _, dice = validate_single_epoch(
            model,
            loss_function,
            dataloaders[TaskType.Valid],
            epoch
        )

        time_avg.append(time() - time_start)

        if dice > dice_best:
            dice_best = dice
            dice_best_epoch = epoch
            torch.save(
                model.state_dict(),
                checkpoint_dir / 'best.pth'
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
    torch.load(
        
    )
    test_single_epoch(
        model,
        loss_function,
        dataloaders[TaskType.Test],
        max_tensorboard_image=params["test"]["max_tensorboard_image"]
    )

if __name__ == '__main__':
    main()