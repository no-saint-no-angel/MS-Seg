from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
# from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms_3che import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        self.transforms = Compose([
                RandomCrop(self.args.crop_size),   # 默认的深度是48
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])

    def __getitem__(self, index):

        ct_array = np.load(self.filename_list[index][0])
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        # ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array / self.args.norm_factor

        ct_array = ct_array.astype(np.float32)

        # ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)
        # print('ct_array and seg_array.shape', ct_array.shape, seg_array.shape)
        if self.transforms:  # 这个transforms就是把体数据原始的深度尺寸修改成指定的尺寸，而高度和宽度尺寸在预处理阶段已经完成，
            ct_array,seg_array = self.transforms(ct_array, seg_array)
            # print('ct_array.shape', ct_array.shape)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())