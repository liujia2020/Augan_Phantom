import os
import argparse
import glob
import torch
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks.generator import AnisotropicUNet, StandardUNet3D
# 导入我们的各向异性生成器
from networks.generator import AnisotropicUNet

def parse_args():
    parser = argparse.ArgumentParser(description="AUGAN 3D Testing Engine (Resave Strategy + Hann Window)")
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--name', type=str, default='AUGAN_MVP_01')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--dir_lq', type=str, default='Recon_HQ_33')
    parser.add_argument('--dir_sq', type=str, default='Recon_SQ_75')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--patch_d', type=int, default=256)
    parser.add_argument('--patch_h', type=int, default=64)
    parser.add_argument('--patch_w', type=int, default=64)
    parser.add_argument('--stride_d', type=int, default=128)
    parser.add_argument('--stride_h', type=int, default=32)
    parser.add_argument('--stride_w', type=int, default=32)
    return parser.parse_args()

# ==============================================================================
# 1. 100% 还原的 9 宫格画图逻辑 (死锁 -60 到 0)
# ==============================================================================
def save_paper_fig_9grid(save_path, case_name, model_name, vol_lq, vol_fake, vol_sq):
    D, H, W = vol_lq.shape
    idx_z = 500 if D > 500 else D // 2
    idx_x = 64  if W > 64  else W // 2
    idx_y = 64  if H > 64  else H // 2
    
    ax_lq, ax_fk, ax_sq = vol_lq[idx_z,:,:], vol_fake[idx_z,:,:], vol_sq[idx_z,:,:]
    sa_lq, sa_fk, sa_sq = vol_lq[:,:,idx_x], vol_fake[:,:,idx_x], vol_sq[:,:,idx_x]
    co_lq, co_fk, co_sq = vol_lq[:,idx_y,:], vol_fake[:,idx_y,:], vol_sq[:,idx_y,:]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))
    fig.suptitle(f"Exp: {model_name} | Case: {case_name}", fontsize=22, fontweight='bold', y=0.95)
    
    rows = [("Axial (Z)", [ax_lq, ax_fk, ax_sq]), 
            ("Sagittal (X)", [sa_lq, sa_fk, sa_sq]), 
            ("Coronal (Y)", [co_lq, co_fk, co_sq])]
    titles = ["Input (LQ)", "Generated (HQ)", "Truth (HQ)"]
    
    for r, (row_name, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap='gray', vmin=-60, vmax=0, aspect='auto')
            if r==0: ax.set_title(titles[c], fontsize=18, fontweight='bold')
            if c==0: ax.set_ylabel(row_name, fontsize=18, fontweight='bold')
            ax.axis('off')
            
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# 2. 100% 还原的 3D 汉宁窗 (消除拼接缝隙)
# ==============================================================================
def get_hann_weight(patch_size):
    d, h, w = patch_size
    window_d = np.hanning(d)
    window_h = np.hanning(h)
    window_w = np.hanning(w)
    weight_3d = window_d[:, None, None] * window_h[None, :, None] * window_w[None, None, :]
    weight_3d = np.clip(weight_3d, 1e-5, 1.0)
    return weight_3d.astype(np.float32)

# ==============================================================================
# 3. 100% 还原的滑窗推理 (死锁 -60 到 0 物理场)
# ==============================================================================
def predict_sliding_window(model, input_vol, patch_size, stride, device):
    # 物理极值截断与映射
    norm_min, norm_max = -60.0, 0.0
    input_vol = np.clip(input_vol, norm_min, norm_max)
    img_norm = (input_vol - norm_min) / (norm_max - norm_min) * 2.0 - 1.0
    
    D, H, W = img_norm.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # 智能 Padding
    pad_d = (pd - D % pd) % pd if D % pd != 0 else 0
    pad_h = (ph - H % ph) % ph if H % ph != 0 else 0
    pad_w = (pw - W % pw) % pw if W % pw != 0 else 0
    # 为保证滑窗能覆盖边缘，额外 pad 一个 patch 大小
    img_pad = np.pad(img_norm, ((0, pad_d+pd), (0, pad_h+ph), (0, pad_w+pw)), mode='reflect')
    
    output_vol = np.zeros_like(img_pad)
    weight_map = np.zeros_like(img_pad)
    patch_weight = get_hann_weight(patch_size)
    
    model.eval()
    
    z_steps = list(range(0, img_pad.shape[0] - pd + 1, sd))
    y_steps = list(range(0, img_pad.shape[1] - ph + 1, sh))
    x_steps = list(range(0, img_pad.shape[2] - pw + 1, sw))
    
    with torch.no_grad():
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:
                    patch = img_pad[z:z+pd, y:y+ph, x:x+pw]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    
                    fake_patch = model(patch_tensor)
                    fake_patch = fake_patch.squeeze().cpu().numpy()
                    
                    output_vol[z:z+pd, y:y+ph, x:x+pw] += fake_patch * patch_weight
                    weight_map[z:z+pd, y:y+ph, x:x+pw] += patch_weight
                    
    # 归一化 & 裁回原图尺寸
    weight_map[weight_map == 0] = 1.0
    output_vol /= weight_map
    
    # 反归一化回 -60 到 0
    output_vol = (output_vol + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
    final_vol = output_vol[:D, :H, :W]
    
    return final_vol

# ==============================================================================
# 4. 纯净且防篡改的保存逻辑
# ==============================================================================
def save_nii(data, save_path, affine, header=None):
    if isinstance(data, torch.Tensor):
        data = data.cpu().float().numpy()
    # 强转 float32，防止被保存为整型
    data = data.astype(np.float32)
    # 关键修复：加入 header，彻底堵死 nibabel 自动缩放极值的可能
    img = nib.Nifti1Image(data, affine, header=header)
    nib.save(img, save_path)

def read_nifti_with_info(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    is_transposed = False
    if data.shape[2] > data.shape[0] and data.shape[2] > data.shape[1]:
        data = data.transpose(2, 1, 0)
        is_transposed = True
    return data, img.affine, img.header, is_transposed

def main():
    opt = parse_args()
    device = torch.device(f'cuda:{opt.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    # model_path = os.path.join(expr_dir, f'netG_epoch_{opt.epoch}.pth')
    model_path = os.path.join(expr_dir, 'weights', f'netG_epoch_{opt.epoch}.pth')
    result_dir = opt.results_dir if opt.results_dir else os.path.join(expr_dir, f'test_results_epoch_{opt.epoch}')
    os.makedirs(result_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🚀 全卷测试 (Resave All Strategy + Hann Window): {opt.name}")
    print("="*80)
    
    # model = AnisotropicUNet(input_nc=1, output_nc=1, ngf=64).to(device)
    if opt.netG == 'standard_unet_3d':
        model = StandardUNet3D(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64).to(device)
        print(">>> 已加载: Standard 3D U-Net (Baseline)")
    else:
        model = AnisotropicUNet(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64).to(device)
        print(">>> 已加载: Anisotropic 3D U-Net")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    test_dir_lq = os.path.join(opt.dataroot, opt.dir_lq)
    test_dir_sq = os.path.join(opt.dataroot, opt.dir_sq)
    files_lq = sorted([f for f in os.listdir(test_dir_lq) if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    patch_size = (opt.patch_d, opt.patch_h, opt.patch_w)
    stride = (opt.stride_d, opt.stride_h, opt.stride_w)
    
    for i, file_name in enumerate(files_lq):
        case_name = file_name.replace('.nii.gz', '').replace('.nii', '')
        file_path_lq = os.path.join(test_dir_lq, file_name)
        
        # 智能匹配 Truth 文件
        base_name = case_name.split('_hq')[0].split('_lq')[0].split('_sq')[0]
        matched_sq_files = glob.glob(os.path.join(test_dir_sq, f"{base_name}*.nii*"))
        has_truth = len(matched_sq_files) > 0
        file_path_sq = matched_sq_files[0] if has_truth else None
        
        print(f"\nProcessing [{i+1}/{len(files_lq)}]: {case_name}")
        
        # 读取输入 (拿到最核心的 affine 和 header，保护极值不被篡改)
        vol_lq, affine, header, transposed = read_nifti_with_info(file_path_lq)
        orig_shape = vol_lq.shape
        
        if has_truth:
            vol_sq, _, _, _ = read_nifti_with_info(file_path_sq)
        else:
            vol_sq = np.zeros_like(vol_lq) - 60.0
            print(f"  ⚠️ 未找到对应的 GT 文件，使用全黑背景代替。")
            
        print("  -> Running sliding window inference with Hann weights...")
        
        # 核心推理 (内部已死锁 -60 到 0 且带汉宁窗)
        vol_fake = predict_sliding_window(model, vol_lq, patch_size, stride, device)
        
        # 画对比图
        view_save_path = os.path.join(result_dir, f"{case_name}_{opt.name}_Comparison.png")
        save_paper_fig_9grid(view_save_path, case_name, opt.name, vol_lq, vol_fake, vol_sq)
        
        # 还原维度准备保存
        if transposed:
            print("  -> Restoring shape for saving...")
            vol_fake_save = vol_fake.transpose(2, 1, 0)
            vol_lq_save   = vol_lq.transpose(2, 1, 0)
            vol_sq_save   = vol_sq.transpose(2, 1, 0)
        else:
            vol_fake_save, vol_lq_save, vol_sq_save = vol_fake, vol_lq, vol_sq
            
        print(f"  -> Saving NIfTI files... (Shape: {vol_fake_save.shape})")
        
        # 强行使用同一个 affine 和 header 保存三个文件！绝对不让 nibabel 乱动数据类型！
        save_nii(vol_fake_save, os.path.join(result_dir, f"{case_name}_{opt.name}_Fake.nii"), affine, header)
        save_nii(vol_lq_save,   os.path.join(result_dir, f"{case_name}_{opt.name}_Input.nii"), affine, header)
        if has_truth:
            save_nii(vol_sq_save, os.path.join(result_dir, f"{case_name}_{opt.name}_Truth.nii"), affine, header)
            
    print(f"\n✅ 测试圆满完成！所有严格对齐的文件和对比图已保存至: {result_dir}")

if __name__ == '__main__':
    main()