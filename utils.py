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

def log_orthogonal_views_to_tb(writer, global_step, input_t, fake_t, target_t, spacing=(0.0326, 0.2, 0.2), save_dir=None):
    """
    核心监控函数：截取 3D 矩阵正中心切片，根据物理间距修正长宽比，绘制 3x3 对比图并送入 TensorBoard。
    布局：
        行: XY断面 (横切), ZY断面 (矢状), ZX断面 (冠状)
        列: Input (左), Fake (中), Target (右)
    """
    dz, dy, dx = spacing
    
    # 抽取 Batch 中的第一个样本并转为 numpy: shape [Z, Y, X]
    inp = input_t[0, 0].detach().cpu().numpy()
    fak = fake_t[0, 0].detach().cpu().numpy()
    tgt = target_t[0, 0].detach().cpu().numpy()
    
    Z, Y, X = inp.shape
    cz, cy, cx = Z//2, Y//2, X//2  # 取正中心切片
    
    # matplotlib 绘制设置
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # 计算图像长宽比 (aspect)。imshow 默认把像素当正方形，利用 aspect 参数还原物理真实比例。
    # XY面 (行=Y, 列=X) => aspect = dy / dx = 0.2 / 0.2 = 1.0
    # ZY面 (行=Z, 列=Y) => aspect = dz / dy = 0.0326 / 0.2 = 0.163
    # ZX面 (行=Z, 列=X) => aspect = dz / dx = 0.0326 / 0.2 = 0.163
    
    views = [
        (inp[cz, :, :], fak[cz, :, :], tgt[cz, :, :], dy/dx, "XY (Axial)\nZ_center"),
        (inp[:, :, cx], fak[:, :, cx], tgt[:, :, cx], dz/dy, "ZY (Sagittal)\n X_center"),
        (inp[:, cy, :], fak[:, cy, :], tgt[:, cy, :], dz/dx, "ZX (Coronal)\n Y_center")
    ]
    
    titles = ["Input (LQ)", "Fake (Generated)", "Target (HQ)"]
    
    for row in range(3):
        slice_inp, slice_fak, slice_tgt, aspect_ratio, row_label = views[row]
        
        # 1. Input (左)
        ax = axes[row, 0]
        ax.imshow(slice_inp, cmap='gray', vmin=-1, vmax=1, aspect=aspect_ratio)
        if row == 0: ax.set_title(titles[0], fontsize=14, fontweight='bold')
        ax.set_ylabel(row_label, fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]) # 隐藏坐标轴
        
        # 2. Fake (中)
        ax = axes[row, 1]
        ax.imshow(slice_fak, cmap='gray', vmin=-1, vmax=1, aspect=aspect_ratio)
        if row == 0: ax.set_title(titles[1], fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        
        # 3. Target (右)
        ax = axes[row, 2]
        ax.imshow(slice_tgt, cmap='gray', vmin=-1, vmax=1, aspect=aspect_ratio)
        if row == 0: ax.set_title(titles[2], fontsize=14, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.tight_layout()
    
    # === 规范化的实体图片保存逻辑 ===
    if save_dir is not None:
        # 拼装出绝对路径，存入传入的专属文件夹
        save_path = os.path.join(save_dir, f'epoch_{global_step:03d}_views.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # 渲染画布并推送到 TensorBoard
    writer.add_figure('Orthogonal_Views_Comparison', fig, global_step)
    
    # (原来在这里的 fig.savefig 到根目录，以及重复的 writer.add_figure 已经被彻底删除了！)
    
    plt.close(fig) # 防止内存泄漏