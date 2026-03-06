# 导入基石模块
from data.dataset import UltrasoundDataset
from networks.generator import AnisotropicUNet
from networks.discriminator import Discriminator3D
import utils # 导入我们刚刚写的极其清爽的工具箱
from networks.losses import AnisotropicSSIMLoss, FFTLoss  # 追加导入 FFTLoss
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
    parser.add_argument('--n_epochs', type=int, default=100, help='保持初始学习率的 Epoch 数量')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='学习率线性衰减到0的 Epoch 数量')
    parser.add_argument('--lambda_ssim', type=float, default=10.0, help='各向异性 SSIM 的权重占比') # <--- 加上这一行
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--norm', type=str, default='batch')
    parser.add_argument('--use_dropout', action='store_true')
    parser.add_argument('--use_sn', action='store_true')
    # parser.add_argument('--lambda_ssim', type=float, default=10.0, help='各向异性 SSIM 的权重占比')
    parser.add_argument('--lambda_fft', type=float, default=0.1, help='FFT 频域损失的权重占比') # <--- 新增这一行
    # ======= 新增：将 L1 和 GAN 的权重也暴露给命令行 =======
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 损失的权重占比')
    parser.add_argument('--lambda_gan', type=float, default=1.0, help='GAN 损失的权重占比')
    parser.add_argument('--resume_epoch', type=int, default=0, help='从指定的 epoch 恢复训练，0 表示从头开始')
    # =========================================================
    return parser.parse_args()

def main():
    opt = parse_args()
    
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    tb_log_dir = os.path.join(expr_dir, 'tb_logs')
    nii_probe_dir = os.path.join(expr_dir, 'nii_probes')
    
    # ======= 新增：定义并创建权重的专属文件夹 =======
    weights_dir = os.path.join(expr_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    # ==============================================
    
    # ================= 新增：定义监控图的专属文件夹 =================
    views_dir = os.path.join(expr_dir, 'monitor_views')
    
    os.makedirs(expr_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(nii_probe_dir, exist_ok=True)
    
    # ================= 新增：创建监控图专属文件夹 =================
    os.makedirs(views_dir, exist_ok=True)
    
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
    criterionSSIM = AnisotropicSSIMLoss(window_size=(27, 5, 5)).to(device) # <--- 实例化我们的特制核
    criterionFFT = FFTLoss().to(device)  # <--- 新增实例化
    

    lambda_L1 = opt.lambda_l1
    lambda_GAN = opt.lambda_gan
    lambda_SSIM = opt.lambda_ssim
    lambda_SSIM = opt.lambda_ssim # <--- 引入权重
    lambda_FFT = opt.lambda_fft          # <--- 获取权重参数
    optimizer_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # ================= 核心增量：线性学习率衰减调度器 =================
    def lambda_rule(epoch):
        # PyTorch 的 LambdaLR 传入的 epoch 默认从 0 开始。
        # 逻辑：如果当前跑过的 epoch 小于恒定期，保持 1.0 倍率；超过恒定期后，计算递减比例。
        lr_l = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return max(0.0, lr_l)
    
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)
    # =================================================================

    
    
    global_step = 0
    total_epochs = opt.n_epochs + opt.n_epochs_decay  # 计算总轮数
    print(f">>> AUGAN 引擎点火，开始训练！总计划 Epoch: {total_epochs}")
    start_epoch = 1
    # ================= 新增：断点恢复逻辑 =================
    if opt.resume_epoch > 0:
        print(f">>> 正在从第 {opt.resume_epoch} 轮恢复训练...")
        path_G = os.path.join(weights_dir, f'netG_epoch_{opt.resume_epoch}.pth')
        path_D = os.path.join(weights_dir, f'netD_epoch_{opt.resume_epoch}.pth')
        
        # 1. 加载网络权重
        netG.load_state_dict(torch.load(path_G, map_location=device))
        netD.load_state_dict(torch.load(path_D, map_location=device))
        
        # 2. 修改起始 Epoch
        start_epoch = opt.resume_epoch + 1
        
        # 3. 推进调度器，恢复到断点时的学习率
        for _ in range(opt.resume_epoch):
            scheduler_G.step()
            scheduler_D.step()
            
        # 4. 恢复 global_step，让 TensorBoard 曲线不断层
        global_step = opt.resume_epoch * len(dataloader)
        print(f">>> 恢复成功！接下来的训练将从第 {start_epoch} 轮开始。当前 LR: {optimizer_G.param_groups[0]['lr']:.6f}")
    # =======================================================
    
    # 修复 1：循环上限改为 total_epochs，打通衰减期
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time() # 记录当前 Epoch 开始时间
        netG.train(); netD.train()
        
        # 修复 2：进度条分母显示总轮数
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch}/{total_epochs}]", leave=False)
        
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
            
            # loss_G_GAN = criterionGAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_GAN = criterionGAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G)) * lambda_GAN
            
            loss_G_L1 = criterionL1(fake_hq, targets_hq) * lambda_L1
            loss_G_SSIM = criterionSSIM(fake_hq, targets_hq) * lambda_SSIM # <--- 计算 SSIM Loss
            loss_G_FFT = criterionFFT(fake_hq, targets_hq) * lambda_FFT     # <--- 计算 FFT Loss
            loss_G = loss_G_GAN + loss_G_L1+ loss_G_SSIM + loss_G_FFT
            
            loss_G.backward()
            optimizer_G.step()
            # ================= 高内聚：字典化统一管理监控项 =================
            # 只要把想看的指标塞进这个字典，后面的代码全自动接管！
            log_dict = {
                'D_Total': loss_D.item(),
                # 'G_GAN': loss_G_GAN.item(),
                # 'G_L1': loss_G_L1.item() / lambda_L1 ,
                'G_GAN': loss_G_GAN.item() / lambda_GAN if lambda_GAN > 0 else 0.0,
                'G_L1': loss_G_L1.item() / lambda_L1 if lambda_L1 > 0 else 0.0,
                'G_SSIM': loss_G_SSIM.item() / lambda_SSIM if lambda_SSIM > 0 else 0.0,
                'G_FFT': loss_G_FFT.item() / lambda_FFT if lambda_FFT > 0 else 0.0  # <--- 挂载到字典！
            }

            global_step += 1
            # 1. 自动推送到 TensorBoard
            for k, v in log_dict.items():
                writer.add_scalar(f'Loss/{k}', v, global_step)

            # 2. 自动格式化更新到 tqdm 进度条尾部
            pbar.set_postfix({k: f"{v:.4f}" for k, v in log_dict.items()})
            

        epoch_duration = time.time() - epoch_start_time
        

        utils.log_orthogonal_views_to_tb(writer, epoch, inputs_lq, fake_hq, targets_hq, spacing=(0.0326, 0.2, 0.2), save_dir=views_dir)
        
        if epoch % 5 == 0 or epoch == opt.n_epochs:
            probe_path = os.path.join(nii_probe_dir, f'epoch_{epoch:03d}_pred.nii')
            utils.save_nifti_probe(fake_hq[0:1], probe_path)
            torch.save(netG.state_dict(), os.path.join(weights_dir, f'netG_epoch_{epoch}.pth'))
            torch.save(netD.state_dict(), os.path.join(weights_dir, f'netD_epoch_{epoch}.pth'))
        
        # ================= 修复 4：极其关键的调度器步进与监控 =================
        scheduler_G.step()
        scheduler_D.step()
        current_lr = optimizer_G.param_groups[0]['lr']
        
        # # 打印清爽的单行总结，包含当前的真实 LR
        # print(f"✅ Epoch [{epoch}/{total_epochs}] 结束 | 耗时: {epoch_duration:.2f} 秒 | 最新 L1: {(loss_G_L1.item()/lambda_L1):.4f} | 当前 LR: {current_lr:.6f}")
        # 3. 自动生成单行日志总结！
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        print(f"✅ Epoch [{epoch}/{total_epochs}] 结束 | 耗时: {epoch_duration:.2f} 秒 | 当前 LR: {current_lr:.6f} | {loss_str}")
    writer.close()
    print(">>> 训练结束！")

if __name__ == '__main__':
    main()