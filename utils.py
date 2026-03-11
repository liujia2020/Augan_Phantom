import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def save_nifti_probe(tensor, save_path):
    """探针函数：将 PyTorch Tensor 转回 .nii 文件保存"""
    data = tensor.squeeze().detach().cpu().numpy()
    # 按照之前的转置逻辑，转回 X, Y, Z 保存更符合常规医学图像软件
    data = data.transpose(2, 1, 0) 
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, save_path)

def log_orthogonal_views_to_tb(writer, global_step, input_extra_t, input_t, fake_t, target_t, spacing=(0.0362, 0.2, 0.2), save_dir=None):
    """
    核心监控函数：截取 3D 矩阵正中心切片，根据物理间距修正长宽比，绘制 3x3 对比图并送入 TensorBoard。
    布局：
        行: XY断面 (横切), ZY断面 (矢状), ZX断面 (冠状)
        列: Extra (极左), Fake (正中), Target (极右)
    """
    dz, dy, dx = spacing
    
    # 抽取 Batch 中的第一个样本并转为 numpy: shape [Z, Y, X]
    # 👑 核心逻辑：这里我们提取 ext (Extra), fak (Fake), tgt (Target)
    # 故意丢弃 inp (LQ)，从而实现把 Extra 3角度放在最左边！
    ext = input_extra_t[0, 0].detach().cpu().numpy()
    fak = fake_t[0, 0].detach().cpu().numpy()
    tgt = target_t[0, 0].detach().cpu().numpy()
    
    Z, Y, X = ext.shape
    cz, cy, cx = Z//2, Y//2, X//2  # 取正中心切片
    
    # matplotlib 绘制设置
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    views = [
        (ext[cz, :, :], fak[cz, :, :], tgt[cz, :, :], dy/dx, "XY (Axial)\nZ_center"),
        (ext[:, :, cx], fak[:, :, cx], tgt[:, :, cx], dz/dy, "ZY (Sagittal)\n X_center"),
        (ext[:, cy, :], fak[:, cy, :], tgt[:, cy, :], dz/dx, "ZX (Coronal)\n Y_center")
    ]
    
    # 明确顶部的标题
    titles = ["Extra (Ref: 3-Angle)", "Fake (Generated)", "Target (HQ: 75-Angle)"]
    
    for row in range(3):
        slice_ext, slice_fak, slice_tgt, aspect_ratio, row_label = views[row]
        
        # 1. Extra (左侧) -> 画的是 ext
        ax = axes[row, 0]
        ax.imshow(slice_ext, cmap='gray', vmin=-1, vmax=1, aspect=aspect_ratio)
        if row == 0: ax.set_title(titles[0], fontsize=14, fontweight='bold')
        ax.set_ylabel(row_label, fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]) 
        
        # 2. Fake (中间) -> 画的是 fak
        ax = axes[row, 1]
        ax.imshow(slice_fak, cmap='gray', vmin=-1, vmax=1, aspect=aspect_ratio)
        if row == 0: ax.set_title(titles[1], fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        
        # 3. Target (右侧) -> 画的是 tgt
        ax = axes[row, 2]
        ax.imshow(slice_tgt, cmap='gray', vmin=-1, vmax=1, aspect=aspect_ratio)
        if row == 0: ax.set_title(titles[2], fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'epoch_{global_step:03d}_views.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    writer.add_figure('Orthogonal_Views_Comparison', fig, global_step)
    plt.close(fig)