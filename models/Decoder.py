import torch.nn as nn


class ConvUpsample3D(nn.Module):
    def __init__(self, in_chans=384, out_chans=[128], upsample=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_tower = nn.ModuleList()
        for i, out_ch in enumerate(self.out_chans):
            if i > 0:
                self.in_chans = out_ch
            # 3D convolution
            self.conv_tower.append(nn.Conv3d(
                self.in_chans, out_ch,
                kernel_size=3, stride=1,
                padding=1, bias=False
            ))
            self.conv_tower.append(nn.GroupNorm(32, out_ch))
            self.conv_tower.append(nn.ReLU(inplace=False))

            # 3D upsampling
            if upsample:
                self.conv_tower.append(nn.Upsample(
                    scale_factor=(2, 2, 2),  # Upsample in D, H, W
                    mode='trilinear',
                    align_corners=False
                ))

        self.convs_level = nn.Sequential(*self.conv_tower)

    def forward(self, x):
        return self.convs_level(x)


class SegmentationHead3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        super().__init__(conv3d)
