import numpy as np

from monai.transforms import Compose
from monai.transforms import (
    Compose, OneOf, RandAffined, Rand2DElasticd, RandGaussianNoised,
    Identity, RandAdjustContrastd,
    RandSimulateLowResolutiond, RandHistogramShiftd, RandGaussianSmoothd,
    RandZoomd, RandFlipd
)


def get_shape_transform(image_keys: list[str], binary_image_keys: list[str]) -> Compose:
    keys = image_keys + binary_image_keys
    mode = list(map(lambda x: "nearest" if x in binary_image_keys else "bilinear", keys))
    align_corners = list(map(lambda x: None if x in binary_image_keys else True, keys))
    return Compose(
        [
            RandZoomd(
                keys=keys,
                min_zoom=0.9,
                max_zoom=1.2,
                mode=mode,
                align_corners=align_corners,
                prob=0.15,
            ),
            RandFlipd(keys, spatial_axis=[0], prob=0.5),
            RandFlipd(keys, spatial_axis=[1], prob=0.5),
        ]
    )

def get_affine_transform(image_keys: list[str], binary_image_keys: list[str]) -> Compose:
    keys = image_keys + binary_image_keys
    mode = list(map(lambda x: "nearest" if x in binary_image_keys else "bilinear", keys))
    return OneOf(
        [
            RandAffined(
                keys=keys,
                mode=mode,
                prob=1.0,
                spatial_size=(512, 512),
                translate_range=20,
                rotate_range=np.pi / 12,
                scale_range=0.15,
                padding_mode="zeros",
            ),
            Rand2DElasticd(
                keys=keys,
                mode=mode,
                prob=1.0,
                spacing=(20, 20),
                magnitude_range=(0.5, 2),
                spatial_size=(512, 512),
                translate_range=20,
                rotate_range=np.pi / 12,
                scale_range=0.15,
                padding_mode="zeros",
            ),
            Identity(),
        ],
        weights=[0.4, 0.4, 0.2],
    )

def get_intensity_transform(keys: list[str]) -> Compose:
    return OneOf(
        [
            RandAdjustContrastd(
                keys=keys, 
                gamma=(0.7, 1.5)
            ),
            RandHistogramShiftd(
                keys=keys, 
                num_control_points=(5, 10)
            ),
            Identity(),
        ],
        weights=[0.25, 0.25, 0.5],
    )

def get_degradation_transform(keys: list[str]) -> Compose:
    return OneOf(
        [
            RandGaussianNoised(keys=keys, prob=1.0, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=keys,
                prob=1.0,
                sigma_x=(0.5, 2),
                sigma_y=(0.5, 2),
                approx="erf",
            ),
            RandSimulateLowResolutiond(
                keys=keys,
                zoom_range=(0.5, 1.0),
            ),
            Identity(),
        ],
        weights=[0.25, 0.25, 0.25, 0.25],
    )