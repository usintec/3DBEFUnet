flowchart TD
  Patches[Patchify & linear proj -> tokens (B, N, C)] --> L1[BasicLayer3D: depth blocks]
  L1 --> WindowAttn[Windowed 3D Self-Attention (local windows)]
  WindowAttn --> Shift[Window Shift (alternate blocks) -> cross-window]
  Shift --> MLP[MLP + Residuals]
  MLP --> PatchMerge[PatchMerging3D -> downsample tokens (B, N/8, 2C)]
