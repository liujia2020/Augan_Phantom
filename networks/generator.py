import torch
import torch.nn as nn

class AnisotropicUNet(nn.Module):
    """
    AUGAN 专属 3D U-Net (极简解耦版)
    - 物理铁律：输入为 (Z=256, Y=64, X=64)。Z轴高频信息极其重要。
    - 架构设计：前 4 层使用各向异性卷积 (保 Z 轴，仅压缩 X/Y)，后 2 层使用各向同性卷积提取全局特征。
    """
    def __init__(self, input_nc=1, output_nc=1, ngf=64):
        super(AnisotropicUNet, self).__init__()

        # === 搭积木过程：从 U-Net 最深处 (Bottleneck) 开始往外包 ===
        
        # 1. 第 6 层 (最内层)：各向同性 (Z, Y, X 一起缩小)
        # 输入: (ngf*8, 64, 2, 2) -> 输出特征图: (ngf*8, 32, 1, 1)
        unet_block = UNetBlock(ngf * 8, ngf * 8, innermost=True, anisotropic=False)
        
        # 2. 第 5 层：各向同性
        # 输入: (ngf*8, 128, 4, 4) -> 输出: (ngf*8, 64, 2, 2)
        unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=unet_block, anisotropic=False)
        
        # 3. 第 4 层：各向异性 (物理红线：从这里开始 Z 轴停止缩小，永远保持 256！)
        # 输入: (ngf*8, 256, 8, 8) -> 输出: (ngf*8, 256, 4, 4)  <-- 注意Z轴维持256
        unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=unet_block, anisotropic=True)
        
        # 4. 第 3 层：各向异性
        # 输入: (ngf*4, 256, 16, 16) -> 输出: (ngf*8, 256, 8, 8)
        unet_block = UNetBlock(ngf * 4, ngf * 8, submodule=unet_block, anisotropic=True)
        
        # 5. 第 2 层：各向异性
        # 输入: (ngf*2, 256, 32, 32) -> 输出: (ngf*4, 256, 16, 16)
        unet_block = UNetBlock(ngf * 2, ngf * 4, submodule=unet_block, anisotropic=True)
        
        # 6. 第 1 层 (最外层)：各向异性
        # 输入图片: (1, 256, 64, 64) -> 输出: (ngf*2, 256, 32, 32)
        self.model = UNetBlock(output_nc, ngf * 2, input_nc=input_nc, submodule=unet_block, outermost=True, anisotropic=True)

    def forward(self, x):
        return self.model(x)


class UNetBlock(nn.Module):
    """
    U-Net 的基础积木块
    包含: DownConv -> [Submodule] -> UpConv -> Concat(Skip Connection)
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, 
                 outermost=False, innermost=False, anisotropic=True):
        super(UNetBlock, self).__init__()
        self.outermost = outermost
        
        if input_nc is None:
            input_nc = outer_nc

        # [核心扩展槽]：这里通过布尔值控制物理维度
        if anisotropic:
            # 各向异性：Z不变(stride=1)，Y和X减半(stride=2)
            k_size = (3, 4, 4)
            s_size = (1, 2, 2)
            p_size = (1, 1, 1)
        else:
            # 各向同性：Z, Y, X 全部减半
            k_size = (4, 4, 4)
            s_size = (2, 2, 2)
            p_size = (1, 1, 1)

        # --- 1. Down 降采样 ---
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=k_size, stride=s_size, padding=p_size, bias=False)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = nn.BatchNorm3d(inner_nc)

        # --- 2. Up 上采样 ---
        uprelu = nn.ReLU(inplace=True)
        upnorm = nn.BatchNorm3d(outer_nc)
        
        # 如果不是最内层，上采样时需要把 Skip Connection 拼进来，所以输入通道是 inner_nc * 2
        upconv_in_nc = inner_nc if innermost else inner_nc * 2
        upconv = nn.ConvTranspose3d(upconv_in_nc, outer_nc, kernel_size=k_size, stride=s_size, padding=p_size, bias=False)

        # --- 3. 灵活组装 ---
        if outermost:
            # 最外层不加 BatchNorm，最后一层使用 Tanh 映射到 [-1, 1]
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=k_size, stride=s_size, padding=p_size, bias=True)
            self.model = nn.Sequential(downconv, submodule, uprelu, upconv, nn.Tanh())
            
        elif innermost:
            # 最内层没有 submodule，不需要拼接 Skip Connection
            self.model = nn.Sequential(downrelu, downconv, uprelu, upconv, upnorm)
            
        else:
            # 标准中间层：降采样 -> 子模块 -> 上采样
            self.model = nn.Sequential(downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # [Skip Connection]: 将当前层的输入 x 与经过 U-Net 腹部处理后的特征在 Channel 维度拼接
            return torch.cat([x, self.model(x)], dim=1)