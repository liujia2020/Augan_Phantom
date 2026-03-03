import os
import random
import numpy as np
import torch
import nibabel as nib
import torch.utils.data as data

class UltrasoundDataset(data.Dataset):
    """
    AUGAN 专属数据集模块 (完全独立版)
    - 移除了冗余的 BaseDataset 继承
    - 移除了晦涩的 opt 传参，参数清晰可见
    - 完美保留核心的物理读取、各向异性裁剪与归一化逻辑
    """
    def __init__(self, dataroot, dir_lq='LQ', dir_sq='HQ', patch_size=(256, 64, 64), norm_min=-1.0, norm_max=1.0):
        super(UltrasoundDataset, self).__init__()
        
        self.dir_lq = os.path.join(dataroot, dir_lq)
        self.dir_sq = os.path.join(dataroot, dir_sq)
        
        # 扫描文件
        self.lq_paths = sorted([os.path.join(self.dir_lq, f) for f in os.listdir(self.dir_lq) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.sq_paths = sorted([os.path.join(self.dir_sq, f) for f in os.listdir(self.dir_sq) if f.endswith('.nii') or f.endswith('.nii.gz')])
        
        # 兼容 tif
        if len(self.lq_paths) == 0:
            self.lq_paths = sorted([os.path.join(self.dir_lq, f) for f in os.listdir(self.dir_lq) if f.endswith('.tif') or f.endswith('.tiff')])
            self.sq_paths = sorted([os.path.join(self.dir_sq, f) for f in os.listdir(self.dir_sq) if f.endswith('.tif') or f.endswith('.tiff')])

        assert len(self.lq_paths) > 0, f"未找到数据! 路径: {self.dir_lq}"
        assert len(self.lq_paths) == len(self.sq_paths), f"文件数量不匹配! LQ={len(self.lq_paths)}, SQ={len(self.sq_paths)}"
            
        print(f"Dataset Initialized. Found {len(self.lq_paths)} paired volumes.")

        # 明确解包参数
        self.patch_d, self.patch_h, self.patch_w = patch_size
        self.norm_min = norm_min
        self.norm_max = norm_max

    def _read_volume(self, path):
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            # Z轴优先的转置逻辑 (完全保留)
            if data.shape[2] > data.shape[0] and data.shape[2] > data.shape[1]:
                data = data.transpose(2, 1, 0)
            return data
        else:
            import tifffile as tiff
            return tiff.imread(path).astype(np.float32)

    def __getitem__(self, index):
        index = index % len(self.lq_paths)
        lq_path = self.lq_paths[index]
        sq_path = self.sq_paths[index]

        # 1. 读取
        img_lq = self._read_volume(lq_path)
        img_sq = self._read_volume(sq_path)
        
        d, h, w = img_lq.shape
        
        # 2. 同步随机裁剪 (完全保留原本逻辑)
        d_max = max(0, d - self.patch_d)
        h_max = max(0, h - self.patch_h)
        w_max = max(0, w - self.patch_w)
        
        z = random.randint(0, d_max)
        y = random.randint(0, h_max)
        x = random.randint(0, w_max)
        
        patch_lq = img_lq[z : z + self.patch_d, y : y + self.patch_h, x : x + self.patch_w]
        patch_hq = img_sq[z : z + self.patch_d, y : y + self.patch_h, x : x + self.patch_w]
        
        # Padding
        if patch_lq.shape != (self.patch_d, self.patch_h, self.patch_w):
            patch_lq = self.pad_tensor(patch_lq)
            patch_hq = self.pad_tensor(patch_hq)

        # 3. 极值截断与归一化 (-1 到 1)
        patch_lq = np.clip(patch_lq, self.norm_min, self.norm_max)
        patch_hq = np.clip(patch_hq, self.norm_min, self.norm_max)
        
        rnge = self.norm_max - self.norm_min
        if rnge == 0: rnge = 1.0
        
        patch_lq = (patch_lq - self.norm_min) / rnge * 2.0 - 1.0
        patch_hq = (patch_hq - self.norm_min) / rnge * 2.0 - 1.0
        
        # 4. 转 Tensor
        tensor_lq = torch.from_numpy(patch_lq).unsqueeze(0).float()
        tensor_hq = torch.from_numpy(patch_hq).unsqueeze(0).float()
        
        # 5. 提取文件名作为 case_name
        case_name = os.path.basename(lq_path)

        # 输出完整的字典结构 (兼容旧代码中的所有的 Key 引用)
        return {
            'lq': tensor_lq, 
            'sq': tensor_hq, 
            'hq': tensor_hq, 
            'lq_path': lq_path, 
            'sq_path': sq_path,
            'case_name': case_name 
        }

    def __len__(self):
        return len(self.lq_paths)
        
    def pad_tensor(self, img):
        d, h, w = img.shape
        pad_d = max(0, self.patch_d - d)
        pad_h = max(0, self.patch_h - h)
        pad_w = max(0, self.patch_w - w)
        return np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), 'constant', constant_values=self.norm_min)