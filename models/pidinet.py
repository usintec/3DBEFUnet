import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import Conv3d
from .config import config_model, config_model_converted

class CSAM3D(nn.Module):
    """
    Compact Spatial Attention Module for 3D MRI volumes
    """
    def __init__(self, channels):
        super(CSAM3D, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv3d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv3d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # initialize bias of conv1 to zero
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        """
        x: [B, C, H, W, D]  (3D MRI input)
        """
        y = self.relu1(x)
        y = self.conv1(y)      # [B, mid_channels, H, W, D]
        y = self.conv2(y)      # [B, 1, H, W, D]
        y = self.sigmoid(y)    # normalize mask
        return x * y           # elementwise spatial attention

class CDCM3D(nn.Module):
    """
    Compact Dilation Convolution based Module for 3D MRI volumes
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM3D, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

        # Multi-scale dilated 3D convolutions
        '''
            ⚠️ Note on large dilations in 3D:
            •	Dilation of 11 in 3D = very large receptive field, may blow up memory on full 3D MRI volumes (240×240×155 in BraTS).
            •	You might want smaller values (2, 3, 5, 7) instead of (5,7,9,11) for stability.
        '''
        self.conv2_1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=5, bias=False)
        self.conv2_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=3, padding=7, bias=False)
        self.conv2_3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=5, padding=9, bias=False)
        self.conv2_4 = nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=7, padding=11, bias=False)

        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        """
        x: [B, C, H, W, D]
        """
        x = self.relu1(x)
        x = self.conv1(x)       # [B, out_channels, H, W, D]

        # Multi-scale dilated convolutions
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)

        return x1 + x2 + x3 + x4
    
class MapReduce3D(nn.Module):
    """
    Reduce 3D feature maps into a single edge map
    """
    def __init__(self, channels):
        super(MapReduce3D, self).__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        """
        x: [B, C, H, W, D]
        Output: [B, 1, H, W, D]  (single edge map per volume)
        """
        return self.conv(x)

class PDCBlock3D(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock3D, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv3d(inplane, ouplane, kernel_size=1, padding=0)

        # 3D PDC version (assuming Conv3d-based PDC is defined as Conv3dPDC)
        self.conv1 = Conv3d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv3d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)

        y = self.conv1(x)       # 3D PDC convolution (depthwise)
        y = self.relu2(y)
        y = self.conv2(y)       # pointwise convolution

        if self.stride > 1:
            x = self.shortcut(x)

        y = y + x               # residual connection
        return y

class PDCBlock3D_converted(nn.Module):
    """
    3D version of PDCBlock_converted
    CPDC, APDC -> 3x3x3 convolution
    RPDC -> 5x5x5 convolution
    """
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock3D_converted, self).__init__()
        self.stride = stride

        # Downsampling if stride > 1
        if self.stride > 1:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv3d(inplane, ouplane, kernel_size=1, padding=0)

        # Depthwise convolution (groups=inplane ensures per-channel conv)
        if pdc == 'rd':  # RPDC -> 5x5x5
            self.conv1 = nn.Conv3d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:  # CPDC, APDC -> 3x3x3
            self.conv1 = nn.Conv3d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)

        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv3d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        if self.stride > 1:
            x = self.pool(x)

        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)

        if self.stride > 1:
            identity = self.shortcut(identity)

        y = y + identity
        return y

class PiDiNet3D(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet3D, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            # Change input channels: MRI is often single channel (1), not RGB (3)
            self.init_block = nn.Conv3d(1, self.inplane,
                    kernel_size=init_kernel_size, padding=init_padding, bias=False)
            block_class = PDCBlock3D_converted
        else:
            self.init_block = Conv3d(pdcs[0], 1, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock3D

        # --- Stage 1 ---
        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        # --- Stage 2 ---
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        # --- Stage 3 ---
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        # --- Stage 4 ---
        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        # --- Feature reduction ---
        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM3D(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM3D(self.dil))
                self.conv_reduces.append(MapReduce3D(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM3D(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce3D(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM3D(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce3D(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce3D(self.fuseplanes[i]))

        # --- Final classifier ---
        self.classifier = nn.Conv3d(4, 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('3D PiDiNet initialization done')

    def forward(self, x):
        D, H, W = x.size()[2:]  # input size: (B, C, D, H, W)

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        # Interpolation for 3D
        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (D, H, W), mode="trilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (D, H, W), mode="trilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (D, H, W), mode="trilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (D, H, W), mode="trilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs

def pidinet_tiny(args):
    pdcs = config_model(args.config)
    dil = 8 if args.dil else None
    return PiDiNet3D(20, pdcs, dil=dil, sa=args.sa)

def pidinet_small(args):
    pdcs = config_model(args.config)
    dil = 12 if args.dil else None
    return PiDiNet3D(30, pdcs, dil=dil, sa=args.sa)

def pidinet(args):
    pdcs = config_model(args.config)
    dil = 24 if args.dil else None
    return PiDiNet3D(60, pdcs, dil=dil, sa=args.sa)


## convert pidinet to vanilla 3D CNN
def pidinet_tiny_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return PiDiNet3D(20, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_small_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 12 if args.dil else None
    return PiDiNet3D(30, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 24 if args.dil else None
    return PiDiNet3D(60, pdcs, dil=dil, sa=args.sa, convert=True)
