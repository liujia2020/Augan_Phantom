import torch
import torch.nn as nn

class Discriminator3D(nn.Module):
    """
    AUGAN 专属 3D PatchGAN 判别器
    - 目标：不只看整体结构，更要抓取局部高频的“超声散斑”纹理。
    - 结构：全卷积网络 (FCN)，输出一个 3D 矩阵，局部感受野对应输入图像的一个 Patch。
    """
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        """
        参数:
            input_nc: 输入通道数。对于条件 GAN (Pix2Pix)，通常是输入图和目标图拼接在一起，所以默认传 2。
                      但在最基础的设计中，也可以分开判别，我们这里预留好接口。
            ndf: 第一层卷积的通道数。
            n_layers: 判别器的深度。层数越多，感受野越大（看的散斑区域越大）。3 层是 PatchGAN 的黄金标准。
        """
        super(Discriminator3D, self).__init__()

        # 第一层不加 BatchNorm (GAN 训练的普遍经验，防止早期梯度震荡)
        # kernel_size=4, stride=2 -> 各向同性压缩，快速扩大感受野
        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        
        # 中间层循环搭建
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) # 通道数翻倍：64 -> 128 -> 256 -> 512 (最大)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # 倒数第二层：步长设为 1，进一步提取特征但不降维
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 最后一层：输出单通道概率图 (真/假 预测)
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        
        # 注意：这里我们没有在末尾加 Sigmoid！
        # 因为在现代 GAN 训练中（比如 LSGAN 或者用 BCEWithLogitsLoss），
        # 把 Sigmoid 融在 Loss 函数里计算会更稳定，避免梯度消失。
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)