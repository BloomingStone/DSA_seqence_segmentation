from torch import nn
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


class ResEncUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512]
        self.strides = [1, 2, 2, 2, 2, 2, 2, 2]

        self.encoder = ResidualEncoder(
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

        self.decoder = UNetDecoder(
            self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=[1,1,1,1,1,1,1],
            deep_supervision=True
        )

    def forward(self, x):
        skips = self.encoder(x)
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

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)