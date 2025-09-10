import torch
from torch import nn
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp, Block
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

import torch
import torch.nn.functional as F

def window_partition_3d(x, window_size):
    """
    Partition 3D feature maps into non-overlapping windows.
    Args:
        x: (B, H, W, D, C)
        window_size: tuple (Wh, Ww, Wd)
    Returns:
        windows: (num_windows*B, Wh, Ww, Wd, C)
    """
    B, H, W, D, C = x.shape
    Wh, Ww, Wd = window_size

    # pad so dimensions are divisible by window_size
    pad_h = (Wh - H % Wh) % Wh
    pad_w = (Ww - W % Ww) % Ww
    pad_d = (Wd - D % Wd) % Wd

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))  # pad order: (last_dim,...,first_dim)
        _, H, W, D, _ = x.shape

    # reshape into windows
    x = x.view(
        B,
        H // Wh, Wh,
        W // Ww, Ww,
        D // Wd, Wd,
        C
    )

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, Wh, Ww, Wd, C)
    return windows


def window_reverse_3d(windows, window_size, H, W, D):
    """
    Reverse 3D windows back to feature maps.
    Args:
        windows: (num_windows*B, Wh, Ww, Wd, C)
        window_size: tuple (Wh, Ww, Wd)
        H, W, D: original sizes (before partitioning)
    Returns:
        x: (B, H, W, D, C)
    """
    Wh, Ww, Wd = window_size
    C = windows.shape[-1]

    # padded sizes
    Hp = (H + Wh - 1) // Wh * Wh
    Wp = (W + Ww - 1) // Ww * Ww
    Dp = (D + Wd - 1) // Wd * Wd

    # compute batch size directly
    B = windows.shape[0] // (Hp // Wh * Wp // Ww * Dp // Wd)

    x = windows.view(
        B,
        Hp // Wh, Wp // Ww, Dp // Wd,
        Wh, Ww, Wd, C
    )

    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, C)

    # crop back to original
    x = x[:, :H, :W, :D, :].contiguous()
    return x


def trunc_normal_(tensor, mean=0., std=1.):
    # Minimal fallback if timm isn't available
    with torch.no_grad():
        return tensor.normal_(mean, std)

class WindowAttention3D(nn.Module):
    r"""
    3D Window-based Multi-Head Self Attention (W-MSA) with 3D relative position bias.

    Args:
        dim (int): embedding dimension (C).
        window_size (tuple[int,int,int]): (Wd, Wh, Ww) cubic window size in voxels.
        num_heads (int): number of attention heads.
        qkv_bias (bool): add learnable bias to q,k,v.
        qk_scale (float|None): override default scale of head_dim ** -0.5 if set.
        attn_drop (float): attention dropout.
        proj_drop (float): projection dropout.
    Inputs:
        x: (num_windows * B, N, C), where N = Wd * Wh * Ww
        mask: None or (num_windows, N, N) — for shifted windows
    Outputs:
        x: (num_windows * B, N, C)
    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert len(window_size) == 3, "window_size must be (Wd, Wh, Ww)"
        self.dim = dim
        self.window_size = tuple(window_size)  # (Wd, Wh, Ww)
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        Wd, Wh, Ww = self.window_size
        # 3D relative position bias table: size = (2*Wd-1)*(2*Wh-1)*(2*Ww-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*Wd - 1) * (2*Wh - 1) * (2*Ww - 1), num_heads)
        )

        # Compute pairwise relative position index for each token inside a 3D window
        coords_d = torch.arange(Wd)
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        # coords: (3, Wd, Wh, Ww) with order (d, h, w)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)  # (3, N)

        # relative_coords: (3, N, N) -> then (N, N, 3)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # shift to start from 0
        relative_coords[:, :, 0] += Wd - 1  # d
        relative_coords[:, :, 1] += Wh - 1  # h
        relative_coords[:, :, 2] += Ww - 1  # w

        # map 3D offsets to a single index
        relative_coords[:, :, 0] *= (2*Wh - 1) * (2*Ww - 1)
        relative_coords[:, :, 1] *= (2*Ww - 1)
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        # projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        x: (B_* = num_windows*B, N, C)  with N = Wd*Wh*Ww
        mask: None or (num_windows, N, N) for SW-MSA3D
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, nH, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, nH, N, N)

        # add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()  # (nH, N, N)
        attn = attn + bias.unsqueeze(0)  # (B_, nH, N, N)

        if mask is not None:
            # mask: (nW, N, N)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # broadcast over batch & heads
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # FLOPs for one window (N = Wd*Wh*Ww)
        flops = 0
        flops += N * self.dim * 3 * self.dim                # qkv proj
        flops += self.num_heads * N * (self.dim//self.num_heads) * N  # q@k^T
        flops += self.num_heads * N * N * (self.dim//self.num_heads)  # attn@v
        flops += N * self.dim * self.dim                    # final proj
        return flops

class SwinTransformerBlock3D(nn.Module):
    r""" Swin Transformer Block for 3D MRI volumes.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): (H, W, D).
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size (Wh, Ww, Wd).
        shift_size (tuple[int]): Shift size (Sh, Sw, Sd).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, dim, input_resolution, num_heads,
                 window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W, D)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Ensure window size fits input
        if min(self.input_resolution) <= min(self.window_size):
            self.shift_size = (0, 0, 0)
            self.window_size = tuple(min(i) for i in zip(self.input_resolution, self.window_size))

        for s, w in zip(self.shift_size, self.window_size):
            assert 0 <= s < w, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # Attention mask for shifted windows
        if any(s > 0 for s in self.shift_size):
            H, W, D = self.input_resolution
            img_mask = torch.zeros((1, H, W, D, 1))  # 1 H W D 1

            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            d_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1

            mask_windows = window_partition_3d(img_mask, self.window_size)  # (nW, Wh, Ww, Wd, 1)
            mask_windows = mask_windows.view(-1, self.window_size[0] *
                                                  self.window_size[1] *
                                                  self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                                   .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W, D = self.input_resolution
        B, L, C = x.shape
        assert L == H * W * D, f"Input feature has wrong size: expected {H*W*D}, got {L}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, D, C)

        # cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_3d(shifted_x, self.window_size)  # (nW*B, Wh, Ww, Wd, C)
        x_windows = x_windows.view(
            -1,
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            C
        )

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (nW*B, N, C)

        # merge windows
        attn_windows = attn_windows.view(
            -1,
            self.window_size[0],
            self.window_size[1],
            self.window_size[2], C
        )
        shifted_x = window_reverse_3d(attn_windows, self.window_size, H, W, D)  # (B, H, W, D, C)

        # reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        # dynamically infer spatial dims
        H_new, W_new, D_new = x.shape[1:4]
        x = x.view(B, H_new * W_new * D_new, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

############ DLF ############
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class MultiScaleBlock3D(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches

        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        inp = x
        
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(inp, self.projs)]

        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], inp[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, inp[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
            
        outs_b = [block(x_) for x_, block in zip(outs, self.blocks)]
        return outs

class BasicLayer3D(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=(0, 0, 0) if (i % 2 == 0) else tuple(w // 2 for w in window_size),
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

class PatchMerging3D(nn.Module):
    r""" 3D Patch Merging Layer for volumetric data (MRI, CT, etc.)
    Args:
        input_resolution (tuple[int]): Resolution of input feature (H, W, D).
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W, D)
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """
        x: B, H*W*D, C
        """
        H, W, D = self.input_resolution
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and D % 2 == 0, f"x size ({H},{W},{D}) not all even."

        x = x.view(B, H, W, D, C)

        # Split into 8 groups
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        # Concatenate along channel dim → B H/2 W/2 D/2 8*C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)  # flatten spatial dims

        # Normalize + Linear projection
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W, D = self.input_resolution
        flops = H * W * D * self.dim
        flops += (H // 2) * (W // 2) * (D // 2) * 8 * self.dim * 2 * self.dim
        return flops

############ Test ############
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

            B, C, H, W = input.shape
            input = input.expand(B, 1, H, W)

            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
