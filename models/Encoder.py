from models.config import config_model
from models.pidinet import PiDiNet3D
import torch
import torch.nn as nn
from utils import *
# from einops import rearrange
from einops.layers.torch import Rearrange
from utils import BasicLayer3D, PatchMerging3D, trunc_normal_


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class SwinTransformer3D(nn.Module):
    def __init__(self, img_size=(128, 128, 128), patch_size=4, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):
        
        super().__init__()
        
        patches_resolution = [img_size[0] // patch_size,
                              img_size[1] // patch_size,
                              img_size[2] // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer3D(  # <<< needs 3D version
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer),
                                  patches_resolution[2] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None  # will need PatchMerging3D
            )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

# --- 3D Pyramid Features ---
class PyramidFeatures3D(nn.Module):
    def __init__(self, config, depth, height, width):
        super().__init__()

        # Swin backbone (3D)
        self.swin_transformer = SwinTransformer3D(
            (depth, height, width), in_chans=1  # MRI → usually single-channel
        )

        # PiDiNet backbone (still 2D → needs true 3D version if you want volumetric convs)
        pidinet = PiDiNet3D(30, config_model(config.pdcs), dil=12, sa=True).eval()

        # load PDC weights
        checkpoint_PDC = torch.load(config.PDC_pretrained_path)
        state_dict = checkpoint_PDC["state_dict"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        pidinet.load_state_dict(new_state_dict)

        self.pidinet_layers = nn.ModuleList(pidinet.children())[:17]

        # 3D channel projection layers
        self.p1_ch = nn.Conv3d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1, stride=4)
        self.p2_ch = nn.Conv3d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1, stride=4)
        self.p3_ch = nn.Conv3d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], kernel_size=1, stride=4)
        self.p4_ch = nn.Conv3d(config.cnn_pyramid_fm[3], config.swin_pyramid_fm[3], kernel_size=1, stride=4)

        # 3D patch merging
        self.p1_pm = PatchMerging3D((depth//4, height//4, width//4), config.swin_pyramid_fm[0])
        self.p2_pm = PatchMerging3D((depth//8, height//8, width//8), config.swin_pyramid_fm[1])
        self.p3_pm = PatchMerging3D((depth//16, height//16, width//16), config.swin_pyramid_fm[2])

        # norms + pooling
        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[3])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # ----- Level 1 -----
        for i in range(4):
            x = self.pidinet_layers[i](x)
        fm1 = x
        fm1_ch = self.p1_ch(fm1)
        fm1_reshaped = fm1_ch.flatten(2).transpose(1, 2)   # (B, DHW, C)
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_skipped = fm1_reshaped + sw1
        norm1 = self.norm_1(sw1_skipped)
        sw1_cls = self.avgpool_1(norm1.transpose(1, 2))
        sw1_cls = Rearrange("b c 1 -> b 1 c")(sw1_cls)
        fm1_sw1 = self.p1_pm(sw1_skipped)

        # ----- Level 2 -----
        fm1_sw2 = self.swin_transformer.layers[1](fm1_sw1)
        for i in range(4, 8):
            fm1 = self.pidinet_layers[i](fm1)
        fm2_ch = self.p2_ch(fm1)
        fm2_reshaped = fm2_ch.flatten(2).transpose(1, 2)
        fm2_sw2_skipped = fm2_reshaped + fm1_sw2
        fm2_sw2 = self.p2_pm(fm2_sw2_skipped)

        # ----- Level 3 -----
        fm2_sw3 = self.swin_transformer.layers[2](fm2_sw2)
        for i in range(8, 12):
            fm1 = self.pidinet_layers[i](fm1)
        fm3_ch = self.p3_ch(fm1)
        fm3_reshaped = fm3_ch.flatten(2).transpose(1, 2)
        fm3_sw3_skipped = fm3_reshaped + fm2_sw3
        fm3_sw3 = self.p3_pm(fm3_sw3_skipped)

        # ----- Level 4 -----
        fm3_sw4 = self.swin_transformer.layers[3](fm3_sw3)
        for i in range(12, 16):
            fm1 = self.pidinet_layers[i](fm1)
        fm4_ch = self.p4_ch(fm1)
        fm4_reshaped = fm4_ch.flatten(2).transpose(1, 2)
        fm4_sw4_skipped = fm4_reshaped + fm3_sw4
        norm2 = self.norm_2(fm4_sw4_skipped)
        sw4_cls = self.avgpool_2(norm2.transpose(1, 2))
        sw4_cls = Rearrange("b c 1 -> b 1 c")(sw4_cls)

        return [
            torch.cat((sw1_cls, sw1_skipped), dim=1),
            torch.cat((sw4_cls, fm4_sw4_skipped), dim=1),
        ]

# 3D MSF Module
class All2Cross3D(nn.Module):
    def __init__(self, config, img_size=(128,128,128), in_chans=1, embed_dim=(96, 768), norm_layer=nn.LayerNorm):
        """
        Args:
            config: model configuration
            img_size (tuple): (D, H, W) of the input MRI volume
            in_chans (int): number of input channels (1 for MRI)
            embed_dim (tuple): embedding dimensions for each branch
        """
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.pyramid = PyramidFeatures3D(config=config, img_size=img_size, in_channels=in_chans)  # <-- upgraded PyramidFeatures3D

        D, H, W = img_size
        patch_d, patch_h, patch_w = config.patch_size  # must be a tuple (d,h,w)

        # number of patches in 3D
        n_p1 = (D // patch_d) * (H // patch_h) * (W // patch_w)
        n_p2 = (D // (patch_d*8)) * (H // (patch_h*8)) * (W // (patch_w*8))
        num_patches = (n_p1, n_p2)

        self.num_branches = 2

        # 3D positional embeddings
        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) 
            for i in range(self.num_branches)
        ])

        # stochastic depth
        total_depth = sum([sum(x[-2:]) for x in config.depth])
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]
        dpr_ptr = 0

        self.blocks = nn.ModuleList()
        for idx, block_config in enumerate(config.depth):
            curr_depth = max(block_config[:-1]) + block_config[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock3D(   # <-- must be a 3D variant
                embed_dim, num_patches, block_config,
                num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                drop_path=dpr_, norm_layer=norm_layer
            )
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        xs = self.pyramid(x)  # -> list of feature maps from PyramidFeatures3D

        if self.cross_pos_embed:
            for i in range(self.num_branches):
                xs[i] += self.pos_embed[i]

        for blk in self.blocks:
            xs = blk(xs)

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        return xs
