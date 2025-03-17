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

from src.core.utils.task_type import TaskType
from src.core.base_augment_transforms import (
    get_affine_transform, get_degradation_transform,
    get_intensity_transform, get_shape_transform
)
from . import params, ImageType


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

def get_data_augment_transform() -> Compose:
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

def get_transforms() -> dict[TaskType, Compose]:
    return {
        TaskType.Train: Compose([
            get_basic_transform(),
            get_data_augment_transform(),
        ]),
        TaskType.Valid: get_basic_transform(),
        TaskType.Test: get_basic_transform(),
    }

def get_data_paths(fold: int) -> dict[TaskType, dict[ImageType, list[Path]]]:
    config = params["dataset"]
    image_dir = Path(config["image_dir"])
    label_dir = Path(config["label_dir"])

    csv_df = pd.read_csv(config["data_info_csv_path"])
    skf = StratifiedKFold(
        n_splits=config["n_splits"], 
        shuffle=True, 
        random_state=config["split_random_seed"]
    )

    train_index, valid_index = list(skf.split(csv_df, csv_df["has_NG"]))[fold]
    train_name_list = csv_df.iloc[train_index]["name"].tolist()
    valid_name_list = csv_df.iloc[valid_index]["name"].tolist()

    return {
        TaskType.Train: {
            ImageType.Image: [image_dir / f'{name}.npy' for name in train_name_list],
            ImageType.Label: [label_dir / f'{name}.npy' for name in train_name_list],
        },
        TaskType.Valid: {
            ImageType.Image: [image_dir / f'{name}.npy' for name in valid_name_list],
            ImageType.Label: [label_dir / f'{name}.npy' for name in valid_name_list],
        },
        TaskType.Test: {
            ImageType.Image: [image_dir / f'{name}.npy' for name in valid_name_list],
            ImageType.Label: [label_dir / f'{name}.npy' for name in valid_name_list],
        },
    }

def get_dataloaders(fold: int) -> dict[TaskType, ThreadDataLoader]:
    data_paths = get_data_paths(fold)
    transforms = get_transforms()

    res: dict[TaskType, ThreadDataLoader] = {}
    for task_type, d in data_paths.items():
        data: list[dict[ImageType, MetaTensor]] = []
        for image_path, label_path in zip(d[ImageType.Image], d[ImageType.Label]):
            image = np.load(image_path)
            label = np.load(label_path)
            first_frame = image[0]

            for i in range(image.shape[0]):
                data.append({
                    ImageType.Image: MetaTensor(image[i], meta={"image_name": image_path.stem, "slice": i}),
                    ImageType.Label: MetaTensor(label[i], meta={"image_name": label_path.stem, "slice": i}),
                    ImageType.FirstImage: MetaTensor(first_frame, meta={"image_name": image_path.stem, "slice": 0}),
                })
                
        res[task_type] = ThreadDataLoader(
            dataset=  CacheDataset(
                data=data,
                transform=transforms[task_type],
            ),
            batch_size=params[task_type]["batch_size"],
            shuffle=params[task_type]["shuffle"],
            num_workers=8,
            pin_memory=True,
        )
    
    return  res


if __name__ == "__main__":
    def _test_dataloader():
        from matplotlib import pyplot as plt
        dataloaders = get_dataloaders(0)
        for dataloader in dataloaders.values():
            batch = next(iter(dataloader))
            for key, value in batch.items():
                print(key, value.shape)
                if value.ndim == 4:
                    data = value[0, 0]
                    image_path = data.meta["image_name"]
                    slice = data.meta["slice"]
                    plt.imshow(data.numpy())
                    plt.show()
                    plt.title(f"{key} - {image_path}- - {slice}")
    _test_dataloader()