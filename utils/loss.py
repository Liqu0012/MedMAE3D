import torch
import torch.nn as nn
from einops import rearrange

class MAE_3D_Loss(nn.Module):
    def __init__(self, patch_embed, in_chans=1, perc_weight=1, norm_pix_loss=True):
        super(MAE_3D_Loss, self).__init__()
        self.patch_embed = patch_embed  # 传入的 patch_embed 实例
        self.in_chans = in_chans
        self.perc_weight = perc_weight
        self.norm_pix_loss = norm_pix_loss

    def patchify3D(self, imgs):
        """
        将输入 3D 图像划分为 patches。
        imgs: (B, C, H, W, D)
        返回: patches (B, L, patch_size**3 * C)
        """
        return rearrange(
            imgs,
            'b c (x p0) (y p1) (z p2) -> b (x y z) (p0 p1 p2 c)',
            p0=self.patch_embed.patch_size[0],
            p1=self.patch_embed.patch_size[1],
            p2=self.patch_embed.patch_size[2]
        )

    def unpatchify3D(self, patches, img_shape):
        """
        将 patches 还原为 3D 图像。
        patches: (B, L, patch_size**3 * C)
        返回: imgs (B, C, H, W, D)
        """
        return rearrange(
            patches,
            'b (h w d) (p0 p1 p2 c) -> b c (h p0) (w p1) (d p2)',
            p0=self.patch_embed.patch_size[0],
            p1=self.patch_embed.patch_size[1],
            p2=self.patch_embed.patch_size[2],
            c=self.in_chans,
            h=self.patch_embed.grid_size[0],
            w=self.patch_embed.grid_size[1],
            d=self.patch_embed.grid_size[2]
        )

    def compute_mse_loss(self, pred, target):
        """
        计算所有patch的MSE损失。
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        return ((pred - target) ** 2).mean()

    def compute_perceptual_loss(self, pred_img, target_img):
        """
        计算感知损失。
        pred_img, target_img: 3D 图像 (B, C, H, W, D)
        这里可以使用一个简单的L1感知损失，也可以扩展为更复杂的感知模型。
        """
        return torch.abs(pred_img - target_img).mean()

    def forward_loss(self, imgs, pred):
        """
        计算综合损失。
        imgs: 原始输入图像 (B, C, H, W, D)
        pred: 模型预测结果 (B, L, patch_size**3 * C)
        """
        # 将输入 3D 图像划分为 patches
        target = self.patchify3D(imgs)  # [B, L, D]
        
        # 计算MSE损失
        mse_loss = self.compute_mse_loss(pred, target)

        # 还原到图像维度计算感知损失
        target_img = imgs  # 使用原始图像作为目标
        pred_img = self.unpatchify3D(pred, imgs.shape)
        perc_loss = self.compute_perceptual_loss(pred_img, target_img)
        # 综合损失
        total_loss = mse_loss + self.perc_weight * perc_loss
        return total_loss

def main():
    # 模拟输入数据
    batch_size = 4
    img_size = (160, 224, 160)
    patch_size = 16  # 单一整数
    in_chans = 1
    embed_dim = 768

    # 模拟输入
    imgs = torch.randn(batch_size, in_chans, *img_size)

    # 初始化 patch embedding
    from patch_embed import PatchEmbed3D
    patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)

    # 初始化损失函数
    loss_fn = MAE_3D_Loss(patch_embed, in_chans=in_chans, norm_pix_loss=True)

    # 模拟模型输出
    num_patches = patch_embed.num_patches
    pred = torch.randn(batch_size, num_patches, patch_size**3 * in_chans)

    # 计算损失
    loss = loss_fn.forward_loss(imgs, pred)
    print(f"Computed Loss: {loss.item()}")

if __name__ == "__main__":
    main()
