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

        # ✅ 3D kernel (Depth × Height × Width)
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
        # ✅ Calls 3D version of PDC
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
            weights_c = weights.sum(dim=[2, 3, 4], keepdim=True)
            yc = F.conv3d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv3d(x, weights, bias, stride=stride, padding=padding,
                         dilation=dilation, groups=groups)
            return y - yc
        return func

    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv3d should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3 and weights.size(4) == 3, \
                'kernel size for ad_conv3d should be 3x3x3'
            assert padding == dilation, 'padding for ad_conv3d set wrong'

            shape = weights.shape  # (out_c, in_c, 3, 3, 3)
            weights = weights.view(shape[0], shape[1], -1)

            # Example reordering for 3D (neighbors vs center) → needs a 3D permutation rule
            # Here I just mimic the 2D clockwise reordering to 3D axial slices
            perm = list(range(weights.size(-1)))
            perm = perm[::-1]  # reverse order as placeholder
            weights_conv = (weights - weights[:, :, perm]).view(shape)

            y = F.conv3d(x, weights_conv, bias, stride=stride,
                         padding=padding, dilation=dilation, groups=groups)
            return y
        return func

    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv3d should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3 and weights.size(4) == 3, \
                'kernel size for rd_conv3d should be 3x3x3'
            padding = 2 * dilation

            shape = weights.shape  # (out_c, in_c, 3, 3, 3)

            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5, 5, 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5, 5, 5)

            weights = weights.view(shape[0], shape[1], -1)

            # Map weights into a 5x5x5 buffer (skeleton, similar to 2D version but extended to 3D)
            # Example: assign edges positive, opposite edges negative, center = 0
            buffer[:, :, 0, :, :] = weights[:, :, :25] * -1  # placeholder mapping
            buffer[:, :, -1, :, :] = weights[:, :, :25]      # adjust as needed
            buffer[:, :, 2, 2, 2] = 0  # center zero

            y = F.conv3d(x, buffer, bias, stride=stride,
                         padding=padding, dilation=dilation, groups=groups)
            return y
        return func

    else:
        print('impossible to be here unless you force that')
        return None
