from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import netron

import transforms
from torchvision import transforms as Tr


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2

        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]

        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


rgc_out_ch = 128

class Basic2(nn.Module):
    def __init__(self):
        super(Basic2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(rgc_out_ch, rgc_out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(rgc_out_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(rgc_out_ch, rgc_out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(rgc_out_ch),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x) + x
        return x


class RGC(nn.Module):
    def __init__(self, channel, ratio=4):
        super(RGC, self).__init__()
        self.basic1 = Basic2()
        self.basic2 = Basic2()
        self.basic3 = Basic2()
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.basic1(x1)
        x2 = self.basic2(x2)
        x2 = self.avg_pool(x2)

        return self.basic3(x2 * x1)


class APRNet(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 0):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])  # encoder_1~6
        self.RGC = RGC(rgc_out_ch)
        self.bn_out = nn.BatchNorm2d(out_ch + 1)
        self.relu = nn.ReLU(inplace=True)

        encode_list = []
        side_list = []
        rgc_pre = []

        # The code related to RGC preprocessing is hidden. If needed, please contact the author.
        # ...




        #


        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                # out_ch==1
                side_list.append(nn.Conv2d(c[3], out_ch + 1, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * (out_ch + 1), out_ch + 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # The code related to encode output, decode output, and side output collecting  are hidden. If needed, please contact the author.
        # ...

        #

        # collect encode outputs
        encode_outputs = []
        encode_outputs_pre = []
        for i, m in enumerate(self.encode_modules):
            # ...



        # collect decode outputs
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            # ...



        # collect side outputs
        side_outputs = []
        for m in self.side_modules:
            # ...



        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs

        else:
            return x


def APRNet_full(out_ch: int = 0):
    cfg = {

        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 1 , 32, 64, False, False],  # En1
                   [6, 64, 32, 128, False, False],  # En2
                   [5, 128, 64, 256, False, False],  # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],  # En5
                   [4, 512, 256, 512, True, True]],  # En6

        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, rgc_out_ch + 512, 256, 512, True, True],  # De5
                   [4, rgc_out_ch + 512, 128, 256, False, True],  # De4
                   [5, rgc_out_ch + 256, 64, 128, False, True],  # De3
                   [6, rgc_out_ch + 128, 32, 64, False, True],  # De2
                   [7, rgc_out_ch + 64, 16, 64, False, True]]  # De1

    }

    return APRNet(cfg, out_ch)


def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 1, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
