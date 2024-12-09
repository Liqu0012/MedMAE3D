# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block
from utils.patch_embed import PatchEmbed3D
from functools import partial
from utils.pos_embed import get_3d_sincos_pos_embed
from utils.loss import MAE_3D_Loss
import timm.models.vision_transformer as timm_vit


class MAE3D(nn.Module):
    def __init__(self, img_size, patch_size=16, in_chans=1, temp_stride=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # 3D-MAE encoder specifics
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 3D-MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3, bias=True)

        # Initialize the loss function with patch_embed
        self.loss_fn = MAE_3D_Loss(self.patch_embed, in_chans=in_chans, norm_pix_loss=True)

        self.initialize_weights()

    def initialize_weights(self):
        # # 3D sin-cos position embedding with xy, xz, and yz planes
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify3D(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L, patch_size**3 *1)
        """
        x = rearrange(imgs, 'b c (x p0) (y p1) (z p2) -> b (x y z) (p0 p1 p2 c)', 
                      p0=self.patch_embed.patch_size[0], p1=self.patch_embed.patch_size[1], p2=self.patch_embed.patch_size[2])
        return x

    def unpatchify3D(self, x):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, H, W, D)
        """
        x = rearrange(x, 'b (h w d) (p0 p1 p2 c) -> b c (h p0) (w p1) (d p2)', 
                    p0=self.patch_embed.patch_size[0], p1=self.patch_embed.patch_size[1], p2=self.patch_embed.patch_size[2], 
                    c=self.in_chans, h=self.patch_embed.grid_size[0], w=self.patch_embed.grid_size[1], d=self.patch_embed.grid_size[2])
        return x

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_3d(self, x, mask_ratio, patch_shape=(4, 4, 4)):
        """
        Randomly mask entire 3D patches with fixed thickness to ensure spatial consistency.
        x: [N, L, D] where L is the number of patches and D is embedding dimension.
        patch_shape: 3D patch size to mask.
        """
        N, L, D = x.shape  # batch, number of patches, embedding dim
        num_patches_to_mask = int(L * mask_ratio)
        
        mask = torch.zeros([N, L], device=x.device)
        
        for i in range(N):
            # Randomly select patches to mask
            masked_indices = torch.randperm(L)[:num_patches_to_mask]
            
            # Apply masking with fixed 3D shape
            for idx in masked_indices:
                start_idx = max(0, idx - patch_shape[0] // 2)
                end_idx = min(L, start_idx + patch_shape[0])
                mask[i, start_idx:end_idx] = 1

        # Mask input
        ids_restore = torch.argsort(mask, dim=1)
        x_masked = x[~mask.bool()].reshape(N, -1, D)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x
    def forward_loss(self, imgs, pred, mask):
        """
        计算最终损失，直接调用外部损失函数。
        imgs: 原始输入图像 (N, C, H, W, D)
        pred: 模型预测结果 (N, num_patches, patch_dim)
        mask: 掩码，表示被遮盖的 patch (N, num_patches)
        """
        # 直接调用 MAE_3D_Loss 实例的 forward_loss 方法
        loss = self.loss_fn.forward_loss(imgs, pred)
        
        return loss
        # target = self.patchify3D(imgs)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5
        # loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # return loss

    def forward(self, imgs, mask_ratio):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        
        if torch.isnan(pred).any():
            print("NaN detected in prediction")
        
        # # 调用外部损失函数，获取每个子损失和总损失
        # total_loss, mse_loss, ssim_loss = self.loss_fn.forward_loss(
        #     pred_patches=pred,
        #     target_patches=self.patchify3D(imgs),
        #     mask_patches=mask,
        #     pred_full=self.unpatchify3D(pred),
        #     target_full=imgs
        # )
        
        # return total_loss, mse_loss, ssim_loss, pred, mask
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae3d_vit_base_patch16_dec512d4b(**kwargs):
    model = MAE3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae3d_vit_large_patch16_dec512d4b(**kwargs):
    model = MAE3D(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae3d_vit_huge_patch14_dec512d4b(**kwargs):
    model = MAE3D(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae3d_vit_base_patch16 = mae3d_vit_base_patch16_dec512d4b  # decoder: 512 dim, 4 blocks
mae3d_vit_large_patch16 = mae3d_vit_large_patch16_dec512d4b  # decoder: 512 dim, 4 blocks
mae3d_vit_huge_patch14 = mae3d_vit_huge_patch14_dec512d4b  # decoder: 512 dim, 4 blocks