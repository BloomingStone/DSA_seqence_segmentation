from pathlib import Path

import torch
from torch import nn
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks
from ...core.base_models.ResEncUNet import ResEncUNet

class ResEncUNet_FirstFrameAssist(ResEncUNet):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.encoder_first_frame = ResidualEncoder(
            input_channels=1,
            n_stages=8,
            features_per_stage=self.features_per_stage,
            conv_op=nn.Conv2d,
            kernel_sizes=3,
            strides=self.strides,
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6, 6],
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"affine": True, "eps": 1e-5},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=True,
            disable_default_stem=False,
        )

        feature_mixtures = []
        for feature in self.features_per_stage:
            feature_mixtures.append(StackedResidualBlocks(
                n_blocks=1,
                conv_op = nn.Conv2d,
                input_channels=2*feature,
                output_channels=feature,
                kernel_size=3,
                initial_stride=1,
                conv_bias=True,
                norm_op=nn.InstanceNorm2d,
                norm_op_kwargs={"affine": True, "eps": 1e-5},
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={"inplace": True},
            ))
        self.feature_mixtures = nn.ModuleList(feature_mixtures)

    def forward(self, x, first_frame):
        skips = self.encoder(x)
        skips_first_frame = self.encoder_first_frame(first_frame)
        for i in range(len(skips)):
            skips[i] = self.feature_mixtures[i](
                torch.cat([skips[i], skips_first_frame[i]], dim=1)
            )

        del skips_first_frame

        out = self.decoder(skips)
        if self.training:
            res = []
            output_shape = out[0].shape
            res.append(out[0])
            for i in range(1, len(out)):
                res.append(nn.functional.interpolate(out[i], output_shape[2:], mode='bilinear', align_corners=True))
            return res
        else:
            return out[0]


def get_model_inited_by_base_model_checkpoints(base_model_checkpoint_path: Path, num_classes: int):
    # 创建新模型实例
    model = ResEncUNet_FirstFrameAssist(num_classes)
    
    # 加载基础模型checkpoint
    checkpoint = torch.load(base_model_checkpoint_path, map_location='cpu', weights_only=True)
    
    # 获取基础模型参数并适配新模型结构
    new_state_dict = {}
    for k, v in checkpoint.items():
        # 复制encoder参数到encoder_first_frame
        if k.startswith('encoder.'):
            new_state_dict[k] = v  # 保留原始encoder参数
            new_state_dict[k.replace('encoder.', 'encoder_first_frame.')] = v  # 复制给encoder_first_frame
        else:
            new_state_dict[k] = v
    
    # 加载参数（strict=False忽略feature_mixtures的新参数）
    model.load_state_dict(new_state_dict, strict=False)
    
    return model


if __name__ == '__main__':
    from . import params
    model = get_model_inited_by_base_model_checkpoints(
        base_model_checkpoint_path=Path(params["model"]['basemodel_checkpoint_path']),
        num_classes=1
    )
    print(model)