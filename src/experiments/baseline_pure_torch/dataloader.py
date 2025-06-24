from pathlib import Path

import numpy as np
from monai.data import DataLoader, CacheDataset # type: ignore
from monai.transforms import (
    LoadImageD,Compose, AsDiscreteD, ClipIntensityPercentilesD, # type: ignore
    EnsureChannelFirstD, EnsureTyped, NormalizeIntensityD, # type: ignore
    SpatialPadD, OneOf, Identity # type: ignore
) 

from ...core.utils.task_type import TaskType
from ...core.base_augment_transforms import (
    get_affine_transform, get_degradation_transform,
    get_intensity_transform, get_shape_transform
)
from .image_type import ImageType

def get_basic_transform() -> Compose:
    return Compose([
        LoadImageD(keys=list(ImageType)),
        EnsureChannelFirstD(keys=list(ImageType), channel_dim="no_channel"),
        SpatialPadD(
            keys=list(ImageType),
            spatial_size=(512, 512),
            method="symmetric",
            mode="constant",
            constant_values = 0
        ),
        EnsureTyped(keys=[ImageType.Image], dtype=np.float32),
        EnsureTyped(keys=[ImageType.Label], dtype=np.int8),
        ClipIntensityPercentilesD(keys=[ImageType.Image], lower=1, upper=99, sharpness_factor=5),
        NormalizeIntensityD(keys=[ImageType.Image], nonzero=True),
        AsDiscreteD(keys=[ImageType.Label], threshold=0.5),
    ])

def get_augment_transform() -> Compose:
    return OneOf(
        [
            Compose([
                get_shape_transform(ImageType.get_image_list(), ImageType.get_binary_image_list()),
                get_affine_transform(ImageType.get_image_list(), ImageType.get_binary_image_list()),
                get_intensity_transform([ImageType.Image]),
                get_degradation_transform([ImageType.Image]),
            ]),
            Identity()
        ],
        weights=[0.8, 0.2],
    )

class DataLoaderManager:
    def __init__(
            self,
            train_image_dir: Path,
            train_label_dir: Path,
            test_image_dir: Path,
            test_label_dir: Path,
            sample_num: int | None = None,
        ):
        self.paths = {
            TaskType.Train: {
                ImageType.Image: sorted(train_image_dir.glob("*.nii.gz")),
                ImageType.Label: sorted(train_label_dir.glob("*.nii.gz")),
            },
            TaskType.Valid: {
                ImageType.Image: sorted(test_image_dir.glob("*.nii.gz")),
                ImageType.Label: sorted(test_label_dir.glob("*.nii.gz")),
            },
        }

        self.basic_transform = get_basic_transform()
        self.augment_transform = get_augment_transform()
        self.sample_num = sample_num

    def get_transform(self, taskType: TaskType) -> Compose:
        if taskType == TaskType.Train:
            return Compose([
                self.basic_transform,
                self.augment_transform,
            ])
        else:
            return self.basic_transform

    def get_dataloader(
            self,
            task_type: TaskType, 
            batch_size: int,
            shuffle: bool,
    ) -> DataLoader:
        if task_type == TaskType.Test:
            task_type = TaskType.Valid
        paths = [
            {
                ImageType.Image: image_path,
                ImageType.Label: label_path,
            }
            for image_path, label_path in zip(
                self.paths[task_type][ImageType.Image],
                self.paths[task_type][ImageType.Label],
            )
        ][:self.sample_num]
        return DataLoader(
            dataset=  CacheDataset(
                data=paths,
                transform=self.get_transform(task_type),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=False,
        )

