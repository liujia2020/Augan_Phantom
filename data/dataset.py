import os
import random
import numpy as np
import torch
import nibabel as nib
import torch.utils.data as data

class UltrasoundDataset(data.Dataset):
    """
    AUGAN 专属数据集模块 (四列对比完全版)
    """
    def __init__(self, dataroot, dir_extra, dir_lq='LQ', dir_sq='HQ', patch_size=(256, 64, 64), norm_min=-1.0, norm_max=1.0):
        super(UltrasoundDataset, self).__init__()
        
        self.dir_lq = os.path.join(dataroot, dir_lq)
        self.dir_sq = os.path.join(dataroot, dir_sq)
        self.dir_extra = os.path.join(dataroot, dir_extra) # 新增：extra 路径
        
        # 扫描文件
        self.lq_paths = sorted([os.path.join(self.dir_lq, f) for f in os.listdir(self.dir_lq) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.sq_paths = sorted([os.path.join(self.dir_sq, f) for f in os.listdir(self.dir_sq) if f.endswith('.nii') or f.endswith('.nii.gz')])
        # 新增：扫描 extra 文件夹
        self.extra_paths = sorted([os.path.join(self.dir_extra, f) for f in os.listdir(self.dir_extra) if f.endswith('.nii') or f.endswith('.nii.gz')])
        
        # 兼容 tif
        if len(self.lq_paths) == 0:
            self.lq_paths = sorted([os.path.join(self.dir_lq, f) for f in os.listdir(self.dir_lq) if f.endswith('.tif') or f.endswith('.tiff')])
            self.sq_paths = sorted([os.path.join(self.dir_sq, f) for f in os.listdir(self.dir_sq) if f.endswith('.tif') or f.endswith('.tiff')])
            self.extra_paths = sorted([os.path.join(self.dir_extra, f) for f in os.listdir(self.dir_extra) if f.endswith('.tif') or f.endswith('.tiff')])

        assert len(self.lq_paths) > 0, f"未找到数据! 路径: {self.dir_lq}"
        assert len(self.lq_paths) == len(self.sq_paths), f"文件数量不匹配! LQ={len(self.lq_paths)}, SQ={len(self.sq_paths)}"
        assert len(self.lq_paths) == len(self.extra_paths), f"Extra文件数量不匹配! Extra={len(self.extra_paths)}"
            
        print(f"Dataset Initialized. Found {len(self.lq_paths)} paired volumes (with extra reference).")

        # 明确解包参数
        self.patch_d, self.patch_h, self.patch_w = patch_size
        self.norm_min = norm_min
        self.norm_max = norm_max
        
    def _read_volume(self, path):
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            if data.shape[2] > data.shape[0] and data.shape[2] > data.shape[1]:
                data = data.transpose(2, 1, 0)
        else:
            import tifffile as tiff
            data = tiff.imread(path).astype(np.float32)

        # 全局归一化
        v_min = data.min()
        v_max = data.max()
        if v_max - v_min > 1e-6:
            data = (data - v_min) / (v_max - v_min) * 2.0 - 1.0
        else:
            data = np.zeros_like(data)
            
        return data

    def __getitem__(self, index):
        index = index % len(self.lq_paths)
        lq_path = self.lq_paths[index]
        sq_path = self.sq_paths[index]
        extra_path = self.extra_paths[index] # 新增：获取 extra 路径

        # 1. 读取
        img_lq = self._read_volume(lq_path)
        img_sq = self._read_volume(sq_path)
        img_extra = self._read_volume(extra_path) # 新增：读取 extra
        
        d, h, w = img_lq.shape
        
        # 2. 同步随机裁剪 (三者使用同一组 z, y, x 坐标)
        d_max = max(0, d - self.patch_d)
        h_max = max(0, h - self.patch_h)
        w_max = max(0, w - self.patch_w)
        
        z = random.randint(0, d_max)
        y = random.randint(0, h_max)
        x = random.randint(0, w_max)
        
        patch_lq = img_lq[z : z + self.patch_d, y : y + self.patch_h, x : x + self.patch_w]
        patch_hq = img_sq[z : z + self.patch_d, y : y + self.patch_h, x : x + self.patch_w]
        patch_extra = img_extra[z : z + self.patch_d, y : y + self.patch_h, x : x + self.patch_w] # 新增：同步裁剪
        
        # 3. 边界 Padding
        if patch_lq.shape != (self.patch_d, self.patch_h, self.patch_w):
            patch_lq = self.pad_tensor(patch_lq)
            patch_hq = self.pad_tensor(patch_hq)
            patch_extra = self.pad_tensor(patch_extra) # 新增：同步 Padding

        # 4. 转 Tensor
        tensor_lq = torch.from_numpy(patch_lq).unsqueeze(0).float()
        tensor_hq = torch.from_numpy(patch_hq).unsqueeze(0).float()
        tensor_extra = torch.from_numpy(patch_extra).unsqueeze(0).float() # 新增：转 Tensor
        
        case_name = os.path.basename(lq_path)

        # 将 extra 打包进返回字典
        return {
            'extra': tensor_extra,
            'lq': tensor_lq, 
            'hq': tensor_hq, 
            'case_name': case_name 
        }
        
    def pad_tensor(self, img):
        d, h, w = img.shape
        pad_d = max(0, self.patch_d - d)
        pad_h = max(0, self.patch_h - h)
        pad_w = max(0, self.patch_w - w)
        return np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), 'constant', constant_values=-1.0)

    def __len__(self):
        return len(self.lq_paths)