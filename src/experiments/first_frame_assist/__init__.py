from pathlib import Path
import tomllib as toml
from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Dict, Any

from src.core.utils.log import init_logger
from src.core.utils.tensorboard import init_writer

class BaseConfig(BaseModel):
    """Base configuration class with common parameters"""
    device: Dict[str, Any] = Field(..., description="Device configuration")
    dataset: Dict[str, Any] = Field(..., description="Dataset configuration")
    model: Dict[str, Any] = Field(..., description="Model configuration")
    train: Dict[str, Any] = Field(..., description="Training parameters")
    test: Dict[str, Any] = Field(..., description="Testing parameters")

class TaskConfig(BaseConfig):
    """Task-specific configuration"""
    @classmethod
    def from_toml(cls, config_path: Path):
        """Load configuration from TOML file"""
        with open(config_path, "rb") as f:
            raw_config = toml.load(f)
        return cls(**raw_config)

def init_task_config(config_path: Path) -> TaskConfig:
    """Initialize task configuration with validation"""
    return TaskConfig.from_toml(config_path)


class ImageType(StrEnum):
    Image = "image"
    Label = "label"
    FirstImage = "first_image"

    @classmethod
    def get_image_list(cls):
        return [cls.Image, cls.FirstImage]

    @classmethod
    def get_binary_image_list(cls):
        return [cls.Label]
