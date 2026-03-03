# 导入基石模块
from data.dataset import UltrasoundDataset
from networks.generator import AnisotropicUNet
from networks.discriminator import Discriminator3D
import utils # 导入我们刚刚写的极其清爽的工具箱

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # <--- 引入进度条神器



def parse_args():
    parser = argparse.ArgumentParser(description="AUGAN 3D Training Engine")
    parser.add_argument('--dataroot', type=str, required=True, help='数据集根目录')
    parser.add_argument('--name', type=str, default='Experiment_01')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--dir_lq', type=str, default='Recon_HQ_33')
    parser.add_argument('--dir_sq', type=str, default='Recon_SQ_75')
    parser.add_argument('--model', type=str, default='augan')
    parser.add_argument('--netG', type=str, default='anisotropic_unet_3d')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size_d', type=int, default=256)
    parser.add_argument('--patch_size_h', type=int, default=64)
    parser.add_argument('--patch_size_w', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--norm', type=str, default='batch')
    parser.add_argument('--use_dropout', action='store_true')
    parser.add_argument('--use_sn', action='store_true')
    return parser.parse_args()

def main():
    opt = parse_args()
    
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    tb_log_dir = os.path.join(expr_dir, 'tb_logs')
    nii_probe_dir = os.path.join(expr_dir, 'nii_probes')
    
    os.makedirs(expr_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(nii_probe_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{opt.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f">>> 实验 [{opt.name}] 初始化完成，设备: {device}")

    # 数据集
    patch_size = (opt.patch_size_d, opt.patch_size_h, opt.patch_size_w)
    dataset = UltrasoundDataset(dataroot=opt.dataroot, dir_lq=opt.dir_lq, dir_sq=opt.dir_sq, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # 网络
    netG = AnisotropicUNet(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64).to(device)
    netD = Discriminator3D(input_nc=opt.input_nc + opt.output_nc, ndf=64).to(device)
    
    # 优化器
    criterionGAN = nn.MSELoss() 
    criterionL1 = nn.L1Loss() 
    lambda_L1 = 100.0
    optimizer_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    global_step = 0
    print(">>> AUGAN 引擎点火，开始训练！")
    
    for epoch in range(1, opt.n_epochs + 1):
        epoch_start_time = time.time() # 记录当前 Epoch 开始时间
        netG.train(); netD.train()
        
        # === 核心改动：用 tqdm 包装 dataloader ===
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch}/{opt.n_epochs}]", leave=False)
        
        for i, batch in pbar:
            inputs_lq = batch['lq'].to(device)
            targets_hq = batch['hq'].to(device)
            fake_hq = netG(inputs_lq)
            
            # --- 训练判别器 D ---
            optimizer_D.zero_grad()
            pred_real = netD(torch.cat((inputs_lq, targets_hq), dim=1))
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
            pred_fake = netD(torch.cat((inputs_lq, fake_hq.detach()), dim=1))
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- 训练生成器 G ---
            optimizer_G.zero_grad()
            pred_fake_for_G = netD(torch.cat((inputs_lq, fake_hq), dim=1))
            loss_G_GAN = criterionGAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_L1 = criterionL1(fake_hq, targets_hq) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            # --- TensorBoard 记录 ---
            global_step += 1
            writer.add_scalar('Loss/D_Total', loss_D.item(), global_step)
            writer.add_scalar('Loss/G_GAN', loss_G_GAN.item(), global_step)
            writer.add_scalar('Loss/G_L1_True', loss_G_L1.item() / lambda_L1, global_step)

            # === 核心改动：将 Loss 实时更新到进度条尾部，不再刷屏打印 ===
            pbar.set_postfix({
                'Loss_D': f"{loss_D.item():.4f}", 
                'Loss_G': f"{loss_G_GAN.item():.4f}", 
                'L1': f"{(loss_G_L1.item()/lambda_L1):.4f}"
            })
        
        # 计算 Epoch 耗时
        epoch_duration = time.time() - epoch_start_time
        
        # 周期性监控与存档
        utils.log_orthogonal_views_to_tb(writer, epoch, inputs_lq, fake_hq, targets_hq, spacing=(0.0326, 0.2, 0.2))
        
        if epoch % 5 == 0 or epoch == opt.n_epochs:
            probe_path = os.path.join(nii_probe_dir, f'epoch_{epoch:03d}_pred.nii')
            utils.save_nifti_probe(fake_hq[0:1], probe_path)
            torch.save(netG.state_dict(), os.path.join(expr_dir, f'netG_epoch_{epoch}.pth'))
            torch.save(netD.state_dict(), os.path.join(expr_dir, f'netD_epoch_{epoch}.pth'))
        
        # 打印清爽的单行总结
        print(f"✅ Epoch [{epoch}/{opt.n_epochs}] 结束 | 耗时: {epoch_duration:.2f} 秒 | 最新 L1: {(loss_G_L1.item()/lambda_L1):.4f}")

    writer.close()
    print(">>> 训练结束！")

if __name__ == '__main__':
    main()