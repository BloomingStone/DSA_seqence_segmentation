import os
from pathlib import Path
from time import time
import tomllib

import torch
from torch import nn
from monai.losses import DiceCELoss     # type: ignore


from .dataloader import DataLoaderManager
from .single_epochs import test_single_epoch
from ...core.base_models.ResEncUNet_pure_torch import ResEncUNet
from ...core.utils.log import init_logger
from ...core.utils.task_type import TaskType


def test(
    task_dir: Path = Path(__file__).parent,
    model: nn.Module = ResEncUNet(output_channels=1),
    params: dict | None = None,
):
    init_logger(task_dir)
    if params is None:
        with open(task_dir / "params.toml", "rb") as f:
            params = tomllib.load(f)
    assert params is not None

    checkpoint_dir = task_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = params["device"]["CUDA_VISIBLE_DEVICES"]
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    model = model.to('cuda')
    loss_function = DiceCELoss(include_background=False, sigmoid=True)

    dataloader_manager = DataLoaderManager(
        train_image_dir=Path(params["dataset"]["train_images_dir"]),
        train_label_dir=Path(params["dataset"]["train_labels_dir"]),
        test_image_dir=Path(params["dataset"]["test_images_dir"]),
        test_label_dir=Path(params["dataset"]["test_labels_dir"]),
    )

    best_checkpoint_path  = task_dir / "checkpoints" /"best.pth"
    test_dataloader = dataloader_manager.get_dataloader(
            TaskType.Test,
            params[TaskType.Test]["batch_size"],
            params[TaskType.Test]["shuffle"]
        )
    model.load_state_dict(torch.load(best_checkpoint_path, map_location='cuda', weights_only=True))
    test_single_epoch(
        model,
        loss_function,
        test_dataloader,
        max_saved_image=params["test"]["max_saved_image"]
    )

if __name__ == '__main__':
    test()