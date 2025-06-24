import os
from pathlib import Path
import tomllib

import torch
from monai.losses import DiceCELoss

from .dataloader import DataLoaderManager
from .model import ResEncUNet_FirstFrameAssist
from .single_epochs import test_single_epoch
from ...core.utils.task_type import TaskType
from ...core.utils.log import init_logger
from ...core.utils.tensorboard import init_writer

_task_dir = Path(__file__).parent

init_logger(_task_dir)
init_writer(_task_dir)

def only_test():
    print('only test')  
    task_dir = Path(__file__).parent.resolve()
    with open(task_dir / "params.toml", "rb") as f:
        params = tomllib.load(f)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params["device"]["CUDA_VISIBLE_DEVICES"]
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    
    model = ResEncUNet_FirstFrameAssist(num_classes=1).to('cuda')
    best_checkpoint_path = task_dir/ 'checkpoints' / 'best.pth'
    model.load_state_dict(torch.load(best_checkpoint_path, map_location='cuda', weights_only=True))
    loss_function = DiceCELoss(sigmoid=True)
    dataloader_manager = DataLoaderManager(
        Path(params["dataset"]["image_dir"]),
        Path(params["dataset"]["label_dir"]),
        Path(params["dataset"]["data_info_csv_path"]),
        params["dataset"]["n_splits"],
        params["dataset"]["split_random_seed"]
    )
    test_single_epoch(
        model,
        loss_function,
        dataloader_manager.get_dataloader(
            TaskType.Test,
            params["dataset"]["fold"],
            params['test']['batch_size'],
            params['test']['shuffle']
        ),
        image_output_dir = task_dir / 'test_output'
    )

if __name__ == '__main__':
    only_test()
    