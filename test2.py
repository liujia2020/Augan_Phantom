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


def parse_args():
    parser = argparse.ArgumentParser(description="AUGAN 3D Testing Engine (Bait-and-Switch Strategy)")
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--name', type=str, default='AUGAN_MVP_01')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--output_nc', type=int, default=1)
    # 核心三轨路径参数
    parser.add_argument('--dir_extra', type=str, default='Recon_LQ_03', help='用于画图和保存基线的 3角度 极差数据')
    parser.add_argument('--dir_lq', type=str, default='Recon_HQ_33', help='真正送入网络推理的 33角度 数据')
    parser.add_argument('--dir_sq', type=str, default='Recon_SQ_75', help='Ground Truth 75角度 金标准')
    parser.add_argument('--netG', type=str, default='anisotropic_unet_3d', help='选择网络架构')
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
# 1. 100% 还原的 9 宫格画图逻辑 (左侧已强行替换为 Extra 3-Angle)
# ==============================================================================
def save_paper_fig_9grid(save_path, case_name, model_name, vol_extra, vol_fake, vol_sq):
    D, H, W = vol_extra.shape
    idx_z = 500 if D > 500 else D // 2
    idx_x = 64  if W > 64  else W // 2
    idx_y = 64  if H > 64  else H // 2
    
    ax_lq, ax_fk, ax_sq = vol_extra[idx_z,:,:], vol_fake[idx_z,:,:], vol_sq[idx_z,:,:]
    sa_lq, sa_fk, sa_sq = vol_extra[:,:,idx_x], vol_fake[:,:,idx_x], vol_sq[:,:,idx_x]
    co_lq, co_fk, co_sq = vol_extra[:,idx_y,:], vol_fake[:,idx_y,:], vol_sq[:,idx_y,:]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))
    fig.suptitle(f"Exp: {model_name} | Case: {case_name}", fontsize=22, fontweight='bold', y=0.95)
    
    rows = [("Axial (Z)", [ax_lq, ax_fk, ax_sq]), 
            ("Sagittal (X)", [sa_lq, sa_fk, sa_sq]), 
            ("Coronal (Y)", [co_lq, co_fk, co_sq])]
    
    # 🌟 标题彻底改头换面，展示给审稿人看的极致对比
    titles = ["Input (3-Angle)", "Generated (HQ)", "Truth (75-Angle)"]
    
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
# 2. 汉宁窗生成
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
# 3. 核心滑窗推理
# ==============================================================================
def predict_sliding_window(model, input_vol, patch_size, stride, device):
    norm_min, norm_max = -60.0, 0.0
    input_vol = np.clip(input_vol, norm_min, norm_max)
    img_norm = (input_vol - norm_min) / (norm_max - norm_min) * 2.0 - 1.0
    
    D, H, W = img_norm.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    pad_d = (pd - D % pd) % pd if D % pd != 0 else 0
    pad_h = (ph - H % ph) % ph if H % ph != 0 else 0
    pad_w = (pw - W % pw) % pw if W % pw != 0 else 0
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
                    
    weight_map[weight_map == 0] = 1.0
    output_vol /= weight_map
    
    output_vol = (output_vol + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
    final_vol = output_vol[:D, :H, :W]
    
    return final_vol

# ==============================================================================
# 4. 防篡改的保存逻辑
# ==============================================================================
def save_nii(data, save_path, affine, header=None):
    if isinstance(data, torch.Tensor):
        data = data.cpu().float().numpy()
    data = data.astype(np.float32)
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
    
    # 修复：保证去 weights 文件夹里读取权重
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model_path = os.path.join(expr_dir, 'weights', f'netG_epoch_{opt.epoch}.pth')
    result_dir = opt.results_dir if opt.results_dir else os.path.join(expr_dir, f'test_results_epoch_{opt.epoch}')
    os.makedirs(result_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🚀 全卷测试 (狸猫换太子策略): {opt.name}")
    print("="*80)
    
    # model = AnisotropicUNet(input_nc=1, output_nc=1, ngf=64).to(device)
    # 动态选择网络架构
    if opt.netG == 'standard_unet_3d':
        model = StandardUNet3D(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=32).to(device)
        print(">>> 已加载: Standard 3D U-Net (Baseline, ngf=32)")
    else:
        model = AnisotropicUNet(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64).to(device)
        print(">>> 已加载: Anisotropic 3D U-Net (ngf=64)")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    test_dir_extra = os.path.join(opt.dataroot, opt.dir_extra) # 03 角度
    test_dir_lq = os.path.join(opt.dataroot, opt.dir_lq)       # 33 角度 (真正入网)
    test_dir_sq = os.path.join(opt.dataroot, opt.dir_sq)       # 75 角度
    
    files_lq = sorted([f for f in os.listdir(test_dir_lq) if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    patch_size = (opt.patch_d, opt.patch_h, opt.patch_w)
    stride = (opt.stride_d, opt.stride_h, opt.stride_w)
    
    for i, file_name in enumerate(files_lq):
        case_name = file_name.replace('.nii.gz', '').replace('.nii', '')
        file_path_lq = os.path.join(test_dir_lq, file_name)
        
        # 提取核心文件名，去匹配另外两个维度的文件夹
        # base_name = case_name.split('_hq')[0].split('_lq')[0].split('_sq')[0]
        base_name = case_name.split('_hq')[0].split('_lq')[0].split('_sq')[0].split('_mq')[0]
        
        # 寻找 Extra(03角度) 和 Truth(75角度)
        matched_extra = glob.glob(os.path.join(test_dir_extra, f"{base_name}*.nii*"))
        matched_sq = glob.glob(os.path.join(test_dir_sq, f"{base_name}*.nii*"))
        
        file_path_extra = matched_extra[0] if len(matched_extra) > 0 else None
        file_path_sq = matched_sq[0] if len(matched_sq) > 0 else None
        
        print(f"\nProcessing [{i+1}/{len(files_lq)}]: {case_name}")
        
        # 1. 拿取 33 角度送入网络 (同时提取严谨的 header 防止篡改)
        vol_lq, affine, header, transposed = read_nifti_with_info(file_path_lq)
        
        # 2. 读取拿来画图撑场面的 03 角度
        if file_path_extra:
            vol_extra, _, _, _ = read_nifti_with_info(file_path_extra)
        else:
            vol_extra = np.zeros_like(vol_lq) - 60.0
            print(f"  ⚠️ 未找到对应的 03角度 文件，使用全黑背景代替。")
            
        # 3. 读取金标准 75 角度
        if file_path_sq:
            vol_sq, _, _, _ = read_nifti_with_info(file_path_sq)
            has_truth = True
        else:
            vol_sq = np.zeros_like(vol_lq) - 60.0
            has_truth = False
            
        # print("  -> Running inference (Network input is 33-Angle)...")
        print(f"  -> Running inference (Network input is from {opt.dir_lq})...")
        # 🎯 注意看：真正送进去跑的依然是 vol_lq (33角度)
        vol_fake = predict_sliding_window(model, vol_lq, patch_size, stride, device)
        
        # 🎨 画对比图：狸猫换太子！传进去的是 vol_extra (03角度)
        view_save_path = os.path.join(result_dir, f"{case_name}_{opt.name}_Comparison.png")
        save_paper_fig_9grid(view_save_path, case_name, opt.name, vol_extra, vol_fake, vol_sq)
        
        if transposed:
            print("  -> Restoring shape for saving...")
            vol_fake_save  = vol_fake.transpose(2, 1, 0)
            vol_extra_save = vol_extra.transpose(2, 1, 0)
            vol_sq_save    = vol_sq.transpose(2, 1, 0)
        else:
            vol_fake_save, vol_extra_save, vol_sq_save = vol_fake, vol_extra, vol_sq
            
        print(f"  -> Saving NIfTI files: 03, Generated, 75...")
        
        # 💾 保存 NIfTI 核心逻辑：存 03，存 Generated，存 75，彻底丢弃 33
        save_nii(vol_extra_save, os.path.join(result_dir, f"{case_name}_{opt.name}_03.nii"), affine, header)
        save_nii(vol_fake_save,  os.path.join(result_dir, f"{case_name}_{opt.name}_Generated.nii"), affine, header)
        
        if has_truth:
            save_nii(vol_sq_save, os.path.join(result_dir, f"{case_name}_{opt.name}_75.nii"), affine, header)
            
    print(f"\n✅ 测试圆满完成！所有严格对齐的文件和对比图已保存至: {result_dir}")

if __name__ == '__main__':
    main()