from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def conv_block(
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1,
        do_act: bool = True
    ):
    blocks = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2),
        nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
    ]
    if do_act:
        blocks.append(nn.LeakyReLU())
    return nn.Sequential(*blocks)

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_block(in_channels, in_channels),
            conv_block(in_channels, in_channels, do_act=False)
        )
        self.act = nn.LeakyReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.blocks(x)
        x += residual
        x = self.act(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, keep_channels: bool):
        super().__init__()
        if keep_channels:
            output_channels = in_channels
            self.bypass = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            output_channels = in_channels * 2
            self.bypass = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            conv_block(in_channels, in_channels * 2, kernel_size=1, stride=1, do_act=False)
        )
        self.blocks = nn.Sequential(
            conv_block(in_channels, output_channels, stride=2),
            conv_block(output_channels, output_channels)
        )
        self.act = nn.LeakyReLU()
        
    
    def forward(self, x: Tensor) -> Tensor:
        residual = self.bypass(x)
        x = self.blocks(x)
        x += residual
        x = self.act(x)
        return x

class UpBlock(nn.Module):
    def __init__(
            self,
            skip_channels: int,
            in_channels: int,
        ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=skip_channels,
            kernel_size=2,
            stride=2
        )
        self.conv = conv_block(skip_channels*2, skip_channels)
    
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.conv_transpose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class Stage(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            keep_channels: bool = False, 
            n_basic_blocks: int = 5
        ):
        super().__init__()
        if keep_channels:
            output_channels = in_channels
        else:
            output_channels = in_channels * 2
        
        blocks: list[nn.Module] = [BasicBlock(output_channels) for _ in range(n_basic_blocks)]

        blocks.insert(0, DownBlock(in_channels, keep_channels))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

class ResEncUNetEncoder(nn.Module):
    def __init__(
            self,
            channels: list[int],
            n_basic_blocks: list[int]
            ):
        assert len(channels) == len(n_basic_blocks) + 1
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(1, channels[0]),
            BasicBlock(channels[0])
        )
        self.channels = channels
        self.keep_channels = [c_in == c_out for c_in, c_out in zip(channels[:-1], channels[1:])]
        self.n_basic_blocks = n_basic_blocks
        self.stages = nn.ModuleList([
            Stage(c, keep_channels=k, n_basic_blocks=n)
            for c, k, n in zip(
                self.channels[:-1], 
                self.keep_channels, 
                self.n_basic_blocks
            )
        ])
    
    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            skips.append(x)
            x = stage(x)
        return x, skips

class ResEncUNetDecoder(nn.Module):
    def __init__(self, encoder: ResEncUNetEncoder, output_channels: int = 2):
        super().__init__()
        up_blocks = []
        out_blocks = []
        channels = encoder.channels[::-1]
        for skip_channel, in_channel in zip(channels[1:], channels[:-1]):
            up_blocks.append(UpBlock(skip_channel, in_channel))
            out_blocks.append(nn.Conv2d(skip_channel, output_channels, kernel_size=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_blocks = nn.ModuleList(out_blocks)
        self.out = nn.Conv2d(channels[-1], output_channels, kernel_size=1)
    
    def forward(self, x: Tensor, skips: list[Tensor]) -> Union[Tensor, list[Tensor]]:
        outs: list[Tensor] = []
        shape = skips[0].shape[-2:]
        for up_block, out_block in zip(self.up_blocks, self.out_blocks):
            x = up_block(x, skips.pop())
            out = out_block(x)
            out = F.interpolate(out, shape, mode='bilinear', align_corners=True)
            outs.append(out)
        x = self.out(x)
        outs.append(x)
        if self.training:
            return outs
        else:
            return outs[-1]

class ResEncUNet(nn.Module):
    def __init__(
            self, 
            output_channels: int = 2,
            channels: list[int] = [32, 64, 128, 256, 512, 512, 512, 512],
            n_basic_blocks: list[int] = [3, 4, 6, 6, 6, 6, 6]
            ):
        super().__init__()
        assert len(channels) == len(n_basic_blocks) + 1
        self.encoder = ResEncUNetEncoder(channels, n_basic_blocks)
        self.decoder = ResEncUNetDecoder(self.encoder, output_channels=output_channels)
    
    def forward(self, x: Tensor) -> Union[Tensor, list[Tensor]]:
        return self.decoder(*self.encoder(x))
