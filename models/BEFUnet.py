import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.Encoder import All2Cross3D
from models.Decoder import ConvUpsample3D, SegmentationHead3D


class BEFUnet3D(nn.Module):
    def __init__(self, config, img_size=(128,128,128), in_chans=4, n_classes=4):
        """
        BEFUnet adapted for 3D MRI brain tumor segmentation.
        Args:
            config: Model configuration object
            img_size: int, assuming cubic volumes (D=H=W=img_size)
            in_chans: int, input channels (1 for MRI modality, >1 if multimodal)
            n_classes: int, number of segmentation classes (e.g., 4 for BraTS: background + WT + TC + ET)
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 32]  # keep patch scaling logic
        self.n_classes = n_classes

        # 3D Encoder
        self.All2Cross = All2Cross3D(config=config, img_size=img_size, in_chans=in_chans)

        # 3D Decoder (conv-upsample)
        self.ConvUp_s = ConvUpsample3D(in_chans=768, out_chans=[128, 128, 128], upsample=True)  # 1
        self.ConvUp_l = ConvUpsample3D(in_chans=96, upsample=False)  # 0

        # Segmentation head (3D)
        self.segmentation_head = SegmentationHead3D(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3
        )

        # 3D prediction conv
        self.conv_pred = nn.Sequential(
            nn.Conv3d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        )
    def forward(self, x):
        xs = self.All2Cross(x)   # list of embeddings [small, large]

        embeddings = [x[:, 1:] for x in xs]  # remove cls token
        reshaped_embed = []

        for i, embed in enumerate(embeddings):
            embed = Rearrange(
                'b (d h w) c -> b c d h w',
                d=(self.img_size[0] // self.patch_size[i]),
                h=(self.img_size[1] // self.patch_size[i]),
                w=(self.img_size[2] // self.patch_size[i])
            )(embed)

            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
            reshaped_embed.append(embed)

        # Fuse multi-scale features
        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        # Final segmentation head
        out = self.segmentation_head(C)

        # return 3 items: seg_logits, embeddings, aux (set aux=None if unused)
        return out, reshaped_embed, None

    # def forward(self, x):
    #     """
    #     Forward pass
    #     Args:
    #         x: Tensor [B, C, D, H, W]
    #     Returns:
    #         Segmentation map [B, n_classes, D, H, W]
    #     """
    #     xs = self.All2Cross(x)   # list of embeddings [small, large]

    #     embeddings = [x[:, 1:] for x in xs]  # remove cls token
    #     reshaped_embed = []

    #     for i, embed in enumerate(embeddings):
    #         # Flatten back to 3D volume tokens
    #         embed = Rearrange(
    #             'b (d h w) c -> b c d h w',
    #             d=(self.img_size[0] // self.patch_size[i]),
    #             h=(self.img_size[1] // self.patch_size[i]),
    #             w=(self.img_size[2] // self.patch_size[i])
    #         )(embed)

    #         # Pass through conv-up
    #         embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
    #         reshaped_embed.append(embed)

    #     # Fuse multi-scale features
    #     C = reshaped_embed[0] + reshaped_embed[1]
    #     C = self.conv_pred(C)

    #     # Final segmentation head
    #     out = self.segmentation_head(C)

    #     return out
