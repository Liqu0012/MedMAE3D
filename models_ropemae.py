import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed

# 引入 RoPE attention 和其他需要的模块
from rope.models.vit_rope import RoPEAttention, RoPE_Layer_scale_init_Block, init_random_2d_freqs, init_t_xy, compute_mixed_cis, compute_axial_cis

class MaskedAutoencoderRoPEViT(nn.Module):
    """Masked Autoencoder with RoPE-enhanced Vision Transformer backbone."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 rope_theta=10.0, rope_mixed=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # RoPE MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 使用RoPE的编码器模块
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            RoPE_Layer_scale_init_Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, Attention_block=RoPEAttention)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # RoPE频率初始化
        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            freqs = [init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta) for _ in range(depth)]
            freqs = torch.stack(freqs, dim=1).view(2, depth, -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x=img_size // patch_size, end_y=img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim // num_heads, theta=rope_theta)
            self.freqs_cis = self.compute_cis(end_x=img_size // patch_size, end_y=img_size // patch_size)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            RoPE_Layer_scale_init_Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, Attention_block=RoPEAttention)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # 将 grid_size 修改为包含高度和宽度的元组
        grid_size = int(self.patch_embed.num_patches**0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (grid_size, grid_size), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (grid_size, grid_size), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # 初始化其他参数
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by shuffling and inserting mask tokens.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch size, sequence length, dimension
        len_keep = int(L * (1 - mask_ratio))

        # 生成随机排列
        noise = torch.rand(N, L, device=x.device)  # 生成范围在 [0, 1] 的随机噪声
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序排序：小的会被保留，大的会被掩盖
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 保留前 `len_keep` 个 token
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # 在掩盖的位置插入“空” tokens
        mask_token = torch.zeros(1, 1, D, device=x.device)  # “空” token
        mask_tokens = mask_token.expand(N, L - len_keep, D)  # 扩展成掩盖 tokens 的数量
        x_full = torch.cat([x_masked, mask_tokens], dim=1)  # 将“空” tokens 添加到序列中
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))  # 恢复顺序

        # 生成二进制掩码：0 表示保留，1 表示掩盖
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_full, mask, ids_restore


    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # 获取输入的实际 patch 数量
        B, L, _ = x.shape  # B: batch size, L: sequence length, _ : embed_dim
        grid_size = int(L**0.5)  # 计算网格大小（假设是方形）

        # 动态计算 `freqs_cis` 以匹配实际输入形状
        if self.rope_mixed:
            # 如果使用混合频率，重新生成 `t_x` 和 `t_y` 以适配实际的 `grid_size`
            t_x, t_y = init_t_xy(end_x=grid_size, end_y=grid_size)
            t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            # 使用轴向频率编码，根据实际 `grid_size` 动态计算 `freqs_cis`
            freqs_cis = compute_axial_cis(dim=self.embed_dim // self.num_heads, end_x=grid_size, end_y=grid_size, theta=self.rope_theta)
            freqs_cis = freqs_cis.to(x.device)

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_tokens = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x, freqs_cis=freqs_cis[i] if self.rope_mixed else freqs_cis)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
def main():
    # 定义模型参数
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 1024
    depth = 24
    num_heads = 16
    decoder_embed_dim = 512
    decoder_depth = 8
    decoder_num_heads = 16
    mlp_ratio = 4.
    norm_pix_loss = False
    rope_theta = 10.0
    rope_mixed = True

    # 实例化 MaskedAutoencoderRoPEViT 模型
    model = MaskedAutoencoderRoPEViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        norm_pix_loss=norm_pix_loss,
        rope_theta=rope_theta,
        rope_mixed=rope_mixed
    )

    # 创建一个随机输入图像数据
    batch_size = 4  # 示例批次大小
    dummy_input = torch.randn(batch_size, in_chans, img_size, img_size)  # [batch, channels, height, width]

    # 设置掩码比例
    mask_ratio = 0.75

    # 执行前向传递
    loss, pred, mask = model(dummy_input, mask_ratio=mask_ratio)

    # 打印输出信息
    print("Loss:", loss.item())
    print("Prediction shape:", pred.shape)
    print("Mask shape:", mask.shape)

if __name__ == "__main__":
    main()