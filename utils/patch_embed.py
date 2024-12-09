# Copyright (c) Cyril Zakka.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Referenced from:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from torch import nn as nn
from torch import _assert

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=1024):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size) 
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.embed_dim = embed_dim

        # 3D卷积操作，用于3D空间补丁划分
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W, D = x.shape  # 输入形状为 (Batch, Channel, Height, Width, Depth)

        # 检查输入尺寸是否符合设定的图像尺寸
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size})."
        
        # 使用3D卷积对输入进行补丁嵌入
        x = self.proj(x)  # 输出形状为 (B, embed_dim, H//patch_H, W//patch_W, D//patch_D)
        x = x.flatten(2).transpose(1, 2)  # 将输出变换为 (B, num_patches, embed_dim) 形状
        return x

# class PatchEmbed3D(nn.Module):
#     """3D Image to Patch Embedding with pre-shuffle Positional Encoding."""
#     def __init__(self, img_size=112, patch_size=16, in_chans=1, embed_dim=768, 
#                  num_frames=16, temp_stride=1, norm_layer=None, flatten=True, add_cls_token=True):
#         super().__init__()
#         img_size = (num_frames, img_size, img_size)
#         patch_size = (temp_stride, patch_size, patch_size)
        
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_frames = num_frames
#         self.embed_dim = embed_dim
#         self.add_cls_token = add_cls_token
        
#         # Calculate grid and number of patches
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
#         self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
#         # 3D Conv to project patches to the embedding dimension
#         self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
#         # CLS token and positional embedding
#         if self.add_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#             self.num_patches += 1  # Adding one for CLS token
        
#         # Position embedding for each patch in the grid before flattening
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#         self.flatten = flatten

#     def forward(self, x):
#         # Verify input shape matches the expected image size
#         _, _, T, H, W = x.shape
#         assert H == self.img_size[1] and W == self.img_size[2] and T == self.img_size[0], \
#             f"Input dimensions ({T}, {H}, {W}) do not match model expected ({self.img_size})"
        
#         # Project patches
#         x = self.proj(x)  # Shape: [B, embed_dim, T', H', W']
        
#         # Flatten and reshape if required
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # Shape: [B, num_patches, embed_dim]
        
#         # Add positional embedding before any shuffling or manipulation
#         x = x + self.pos_embed[:, :x.size(1), :]  # Ensures compatibility with positional encoding size
        
#         # Add CLS token if enabled
#         if self.add_cls_token:
#             cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # Shape: [B, 1, embed_dim]
#             x = torch.cat((cls_tokens, x), dim=1)  # Concatenate along the patch dimension
            
#         # Normalize if needed
#         x = self.norm(x)
        
#         return x
