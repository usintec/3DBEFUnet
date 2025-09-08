import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv3d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv3d, self).__init__()
        
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 3D kernel (Depth × Height × Width)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.pdc = pdc   # pixel-difference convolution function (must also support 3D!)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Calls 3D version of PDC
        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

## cd, ad, rd convolutions (3D)
def createConvFunc3D(op_type):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], f'unknown op type: {op_type}'

    if op_type == 'cv':
        return F.conv3d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv3d should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3 and weights.size(4) == 3, \
                'kernel size for cd_conv3d should be 3x3x3'
            assert padding == dilation, 'padding for cd_conv3d set wrong'

            # channel-wise center (sums over kernel volume)
            # weights shape: (out_c, in_c//groups, 3,3,3) -> keep in-channel dim
            weights_c = weights.sum(dim=[2, 3, 4], keepdim=True)  # (out_c, in_c//groups, 1,1,1)
            # yc: conv with collapsed kernel (no padding)
            yc = F.conv3d(x, weights_c, bias=None, stride=stride, padding=0, dilation=1, groups=groups)
            y = F.conv3d(x, weights, bias, stride=stride,
                         padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func

    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv3d should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3 and weights.size(4) == 3, \
                'kernel size for ad_conv3d should be 3x3x3'
            assert padding == dilation, 'padding for ad_conv3d set wrong'

            shape = weights.shape  # (out_c, in_c//groups, 3, 3, 3)
            weights_flat = weights.view(shape[0], shape[1], -1)  # (out_c, in_c//groups, 27)

            # TODO: implement a principled 3D neighbor permutation for "angular" differences.
            # For now we use a simple placeholder reversal (keeps shape correct).
            perm = list(range(weights_flat.size(-1)))
            perm = perm[::-1]
            weights_conv = (weights_flat - weights_flat[:, :, perm]).view(shape)

            y = F.conv3d(x, weights_conv, bias, stride=stride,
                         padding=padding, dilation=dilation, groups=groups)
            return y
        return func

    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            """
            Radial-difference style conv for 3D.
            This implementation:
              - expects a 3x3x3 kernel (weights)
              - embeds the 3x3x3 kernel into the center of a 5x5x5 buffer
              - zeroes the center voxel (so center contribution is removed)
              - runs conv3d with that 5x5x5 kernel and padding = 2 * dilation
            This is a safe, shape-correct generalisation of the 2D skeleton mapping.
            """
            assert dilation in [1, 2], 'dilation for rd_conv3d should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3 and weights.size(4) == 3, \
                'kernel size for rd_conv3d should be 3x3x3'

            padding = 2 * dilation
            shape = weights.shape  # (out_c, in_c//groups, 3, 3, 3)

            # Build buffer on the same device & dtype as weights
            device = weights.device
            dtype = weights.dtype
            buffer = torch.zeros(shape[0], shape[1], 5, 5, 5, device=device, dtype=dtype)

            # reshape original 3x3x3 kernels
            w3 = weights.view(shape[0], shape[1], 3, 3, 3)

            # place the 3x3x3 kernel into the center of 5x5x5 buffer
            # center coordinates in 5x5x5 are indices 1..3
            buffer[:, :, 1:4, 1:4, 1:4] = w3

            # zero the center voxel to remove direct center response (radial-style)
            buffer[:, :, 2, 2, 2] = 0.0

            # Note: you can replace the above placement with any radial mapping you prefer,
            # for example distributing signs or weights to outer shells.

            y = F.conv3d(x, buffer, bias, stride=stride,
                         padding=padding, dilation=dilation, groups=groups)
            return y
        return func

    else:
        print('impossible to be here unless you force that')
        return None
