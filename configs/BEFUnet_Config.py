import ml_collections
import os

os.makedirs('./weights', exist_ok=True)


# BEFUnet 3D Configs
def get_BEFUnet_configs():
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs (3D)
    cfg.swin_pyramid_fm = [96, 192, 384, 768]  # feature dims
    cfg.image_size = [128, 128, 128]  # [D, H, W] volume size (crop/patch size for training)
    cfg.patch_size = [2, 4, 4]        # patch depth, height, width
    cfg.num_classes = 4               # BraTS: background + 3 tumor classes

    # ⚠️ Pretrained Swin 2D weights are not compatible with 3D
    # You can later adapt or inflate weights if you need
    cfg.swin_pretrained_path = None

    # CNN (PiDiNet3D) Configs
    cfg.cnn_backbone = "pidinet_small_converted"
    cfg.pdcs = 'carv4'
    cfg.cnn_pyramid_fm = [30, 60, 120, 120]
    cfg.pidinet_pretrained = False
    cfg.PDC_pretrained_path = None

    # Dual-Level Fusion (DLF) Configs
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)         # multi-head attention counts
    cfg.mlp_ratio = (2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg
