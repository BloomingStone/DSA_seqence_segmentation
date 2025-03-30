from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from monai.data import ThreadDataLoader, MetaTensor, CacheDataset
from monai.transforms import (
    Compose, AsDiscreted, ClipIntensityPercentilesd,
    EnsureChannelFirstd, EnsureTyped, NormalizeIntensityd,
    SpatialPadd, OneOf, Identity
)

from ...core.utils.task_type import TaskType
from ...core.base_augment_transforms import (
    get_affine_transform, get_degradation_transform,
    get_intensity_transform, get_shape_transform
)
from .image_type import ImageType

def get_basic_transform() -> Compose:
    return Compose([
        EnsureChannelFirstd(keys=list(ImageType), channel_dim="no_channel"),
        SpatialPadd(
            keys=list(ImageType),
            spatial_size=(512, 512),
            method="symmetric",
            mode="constant",
            constant_values = 0
        ),
        EnsureTyped(keys=[ImageType.Image, ImageType.FirstImage], dtype=np.float32),
        EnsureTyped(keys=[ImageType.Label], dtype=np.int8),
        ClipIntensityPercentilesd(keys=[ImageType.Image, ImageType.FirstImage], lower=1, upper=99, sharpness_factor=5),
        NormalizeIntensityd(keys=[ImageType.Image, ImageType.FirstImage], nonzero=True),
        AsDiscreted(keys=[ImageType.Label], threshold=0.5),
    ])

def get_augment_transform() -> Compose:
    return OneOf(
        [
            Compose([
                get_shape_transform(ImageType.get_image_list(), ImageType.get_binary_image_list()),
                get_affine_transform(ImageType.get_image_list(), ImageType.get_binary_image_list()),
                get_intensity_transform([ImageType.Image, ImageType.FirstImage]),
                get_degradation_transform([ImageType.Image, ImageType.FirstImage]),
            ]),
            Identity()
        ],
        weights=[0.8, 0.2],
    )

class DataLoaderManager:
    def __init__(
            self,
            image_dir: Path,
            label_dir: Path,
            data_info_csv_path: Path,
            n_splits: int,
            random_seed: int,
        ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.csv_df = pd.read_csv(data_info_csv_path)

        skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_seed
        )
        self.k_folds = list(skf.split(self.csv_df, self.csv_df["has_NG"]))

        self.basic_transform = get_basic_transform()
        self.augment_transform = get_augment_transform()

    def get_transform(self, taskType: TaskType) -> Compose:
        if taskType == TaskType.Train:
            return Compose([
                self.basic_transform,
                self.augment_transform,
            ])
        else:
            return self.basic_transform

    def get_data_paths(self, task_type, fold: int) -> dict[ImageType, list[Path]]:
        train_index, valid_index = self.k_folds[fold]
        if task_type == TaskType.Train:
            name_list = self.csv_df.iloc[train_index]["name"].tolist()
        else:
            name_list = self.csv_df.iloc[valid_index]["name"].tolist()
        
        return {
            ImageType.Image: [self.image_dir / f'{name}.npy' for name in name_list],
            ImageType.Label: [self.label_dir / f'{name}.npy' for name in name_list],
        }

    def get_dataloader(
            self,
            task_type: TaskType, 
            fold: int,
            batch_size: int,
            shuffle: bool,
    ) -> ThreadDataLoader:
        data_paths = self.get_data_paths(task_type, fold)
        data: list[dict[ImageType, MetaTensor]] = []
        for image_path, label_path in zip(data_paths[ImageType.Image], data_paths[ImageType.Label]):
            image = np.load(image_path)
            label = np.load(label_path)
            first_frame = image[0]

            for i in range(image.shape[0]):
                data.append({
                    ImageType.Image: MetaTensor(image[i], meta={"image_name": image_path.stem, "slice": i}),
                    ImageType.Label: MetaTensor(label[i], meta={"image_name": label_path.stem, "slice": i}),
                    ImageType.FirstImage: MetaTensor(first_frame, meta={"image_name": image_path.stem, "slice": 0}),
                })
                
        return ThreadDataLoader(
            dataset=  CacheDataset(
                data=data,
                transform=self.get_transform(task_type),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
        )
