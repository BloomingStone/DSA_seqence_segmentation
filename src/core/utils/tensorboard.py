from typing import Optional, Callable
from datetime import datetime
from pathlib import Path
from functools import wraps

import torch
from einops import rearrange
from monai.handlers.tensorboard_handlers import SummaryWriter
from torch import Tensor
from torchvision.utils import make_grid

from .task_type import TaskType


_writer: Optional[SummaryWriter] = None

def valid_init(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _writer
        if _writer is None:
            raise RuntimeError("TensorBoard writer not initialized. Call init_writer first.")
        return func(*args, **kwargs)
    return wrapper

def init_writer(root_dir: Path, name: str = None) -> None:
    global _writer
    if _writer is not None:
        return
    
    assert root_dir.exists() and root_dir.is_dir()
    
    if name is None:
        begin_time: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        tensorboard_dir = root_dir / "tensorboard" / begin_time
    else:
        tensorboard_dir = root_dir / "tensorboard" / name
    
    tensorboard_dir.mkdir(exist_ok=True, parents=True)
    _writer = SummaryWriter(log_dir=str(tensorboard_dir))

@valid_init
def record_scalar(task_type: TaskType, name: str, value: float, epoch: int) -> None:
    _writer.add_scalar(f"{task_type.value}/{name}", value, epoch)

@valid_init
def record_learning_rate(optimizer: torch.optim.Optimizer, epoch: int):
    for i, param_group in enumerate(optimizer.param_groups):
        record_scalar(TaskType.Train, f"lr_group_{i}", param_group["lr"], epoch)

@valid_init
def record_image_label_prediction(
    image_name: str,
    image: Tensor,
    label: Tensor,
    prediction: Tensor,
):
    if not (image.dim() == 2 and label.dim() == 2 and prediction.dim() == 2):
        print("Only support 2D image with 1 channel now!")
        return
    
    image = rearrange(image, '(c h) w -> c h w', c=1)
    image = (image - image.min()) / (image.max() - image.min())
    label = rearrange(label, '(c h) w -> c h w', c=1)
    prediction = rearrange(prediction, '(c h) w -> c h w', c=1)

    alpha = 0.4
    overlay_pred = (prediction * alpha + image).clamp(0, 1)
    overlay_gt = (label * alpha + image).clamp(0, 1)

    combined_image = torch.stack([image, overlay_gt, overlay_pred])
    grid_image = make_grid(combined_image)
    _writer.add_image(image_name, grid_image)
