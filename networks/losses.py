import torch
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    """生成一维高斯核"""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window_3d(window_size_z, window_size_y, window_size_x, channel=1):
    """通过一维高斯核的广播机制 (Broadcasting)，生成绝对各向异性的 3D 物理感受野"""
    # 生成三个方向的 1D 张量
    _1D_window_z = gaussian(window_size_z, 1.5).unsqueeze(1).unsqueeze(2) # Shape: [27, 1, 1]
    _1D_window_y = gaussian(window_size_y, 1.5).unsqueeze(0).unsqueeze(2) # Shape: [1, 5, 1]
    _1D_window_x = gaussian(window_size_x, 1.5).unsqueeze(0).unsqueeze(1) # Shape: [1, 1, 5]
    
    # 👑 核心修复：直接利用 PyTorch 的广播机制连乘，抛弃死板的 .mm()
    _3D_window = _1D_window_z * _1D_window_y * _1D_window_x
    
    # 扩展到指定的通道数 (通常是 1)
    window = _3D_window.expand(channel, 1, window_size_z, window_size_y, window_size_x).contiguous()
    return window

class AnisotropicGradientLoss(torch.nn.Module):
    """
    各向异性 3D 梯度损失 (Anisotropic 3D Gradient Loss)
    利用有限差分法提取边缘突变，并强行给予 Z 轴 5.52 倍的物理惩罚权重，
    以完美补偿 Z 轴与 X/Y 轴极其悬殊的物理间距差异 (0.2 / 0.0362 ≈ 5.52)。
    """
    def __init__(self):
        super(AnisotropicGradientLoss, self).__init__()
        # 使用 L1 计算梯度的绝对误差
        self.criterion = torch.nn.L1Loss()
        # 👑 核心物理外挂：Z 轴权重补偿
        self.z_weight = 5.52 

    def forward(self, fake, target):
        # 1. Z 轴差分 (深度方向)
        fake_dz = fake[:, :, 1:, :, :] - fake[:, :, :-1, :, :]
        target_dz = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        loss_z = self.criterion(fake_dz, target_dz) * self.z_weight

        # 2. Y 轴差分 (高度方向)
        fake_dy = fake[:, :, :, 1:, :] - fake[:, :, :, :-1, :]
        target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        loss_y = self.criterion(fake_dy, target_dy)

        # 3. X 轴差分 (宽度方向)
        fake_dx = fake[:, :, :, :, 1:] - fake[:, :, :, :, :-1]
        target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        loss_x = self.criterion(fake_dx, target_dx)

        # 汇总三个维度的梯度误差
        return loss_z + loss_y + loss_x

class AnisotropicSSIMLoss(torch.nn.Module):
    """
    量身定制的各向异性 3D SSIM 损失
    默认尺寸 (Z=27, Y=5, X=5) 精确匹配体素间距 0.0362 * 0.2 * 0.2
    """
    def __init__(self, window_size=(27, 5, 5), channel=1):
        super(AnisotropicSSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window_3d(window_size[0], window_size[1], window_size[2], channel)
        # 根据各向异性核的大小自动计算 padding，保证输出 shape 和输入一致
        self.padding = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)

    def forward(self, img1, img2):
        # 动态匹配数据所在的设备 (GPU/CPU)
        if img1.device != self.window.device:
            self.window = self.window.to(img1.device)
            
        mu1 = F.conv3d(img1, self.window, padding=self.padding, groups=self.channel)
        mu2 = F.conv3d(img2, self.window, padding=self.padding, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, self.window, padding=self.padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, self.window, padding=self.padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, self.window, padding=self.padding, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 深度学习中 Loss 越小越好，因此返回 1 - SSIM
        return 1.0 - ssim_map.mean()

class FFTLoss(torch.nn.Module):
    """
    频域损失 (Frequency Loss / FFT Loss)
    利用 3D 快速傅里叶变换提取振幅谱，强迫网络消除周期性的空间条纹伪影（如百叶窗）。
    """
    def __init__(self):
        super(FFTLoss, self).__init__()
        # 频域振幅的差异，我们使用 L1 Loss 来计算绝对误差
        self.criterion = torch.nn.L1Loss()

    def forward(self, fake, target):
        # 针对 5D 张量 (B, C, Z, Y, X)，在最后三个空间维度 (Z, Y, X) 上进行 3D 傅里叶变换
        fake_fft = torch.fft.rfftn(fake, dim=(2, 3, 4))
        target_fft = torch.fft.rfftn(target, dim=(2, 3, 4))
        
        # 获取频域的振幅谱 (求模: 取复数的绝对值)
        fake_amplitude = torch.abs(fake_fft)
        target_amplitude = torch.abs(target_fft)
        
        # 计算生成图和目标图在频域振幅上的 L1 距离
        return self.criterion(fake_amplitude, target_amplitude)