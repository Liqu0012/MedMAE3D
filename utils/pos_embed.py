# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple (height, width, depth) representing grid dimensions
    return:
    pos_embed: [grid_size[0]*grid_size[1]*grid_size[2], embed_dim] or [1+grid_size[0]*grid_size[1]*grid_size[2], embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)  # Height
    grid_w = np.arange(grid_size[1], dtype=np.float32)  # Width
    grid_d = np.arange(grid_size[2], dtype=np.float32)  # Depth
    grid = np.meshgrid(grid_h, grid_w, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # Add an additional token if cls_token is True
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid, cls_token=False):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[0])  # (H*W*D, D/4)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[1])  # (H*W*D, D/4)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[2])  # (H*W*D, D/4)
    emb_extra = np.zeros((emb_h.shape[0], embed_dim // 4), dtype=np.float32)  # (H*W*D, embed_dim/4)
    
    emb = np.concatenate([emb_h, emb_w, emb_d, emb_extra], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple (height, width) representing grid dimensions
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)  # Height
    grid_w = np.arange(grid_size[1], dtype=np.float32)  # Width
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

if __name__ == '__main__':
    class MAE3D(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, temp_stride=2, norm_layer=nn.LayerNorm):
            super().__init__()
            
            self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=(temp_stride, patch_size, patch_size), stride=(temp_stride, patch_size, patch_size))
            grid_size = (num_frames // temp_stride, img_size // patch_size, img_size // patch_size)
            self.pos_embed_xy, self.pos_embed_xz, self.pos_embed_yz = [nn.Parameter(torch.from_numpy(embed).float(), requires_grad=False) for embed in get_3d_sincos_pos_embed(embed_dim, grid_size)]
            
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12) for _ in range(12)])
            self.norm = norm_layer(embed_dim)
            
        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)  
            B, N, _ = x.shape
            
            # 添加xy、xz、yz嵌入
            pos_embed = self.pos_embed_xy + self.pos_embed_xz + self.pos_embed_yz

            x = x + pos_embed.unsqueeze(0).repeat(B, 1, 1)  
            
            # 添加CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Transformer blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            
            return x