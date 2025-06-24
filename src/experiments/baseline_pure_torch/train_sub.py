from pathlib import Path
import tomllib


from .train import train
from ...core.base_models.ResEncUNet_pure_torch import ResEncUNet

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("sub_experiment_name", type=str, default="sub experiment name")
    args = parser.parse_args()
    sub_experiment_name = args.sub_experiment_name
    
    task_dir = Path(__file__).parent / "sub_experiments" / sub_experiment_name
    assert task_dir.exists(), f"Task directory {task_dir} does not exist"
    
    with open(task_dir / "params.toml", "rb") as f:
        params = tomllib.load(f)
    
    train(
        task_dir=task_dir,
        model=ResEncUNet(
            output_channels=1,
            channels=params["model"]["channels"],
            n_basic_blocks=params["model"]["n_basic_blocks"]
        ),
        params=params
    )