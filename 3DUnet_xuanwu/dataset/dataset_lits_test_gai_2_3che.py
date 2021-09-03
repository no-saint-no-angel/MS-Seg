from torch._C import dtype
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk


"""
    测试的时候把原始data和label文件，在高度和宽度尺寸上都按照(args.slice_down_scale, args.xy_down_scale, args.xy_down_scale)，调整为256*256，
    而对于data文件，由于一整个文件直接训练对显存要求高，按照深度test_cut_size为48，步距test_cut_stride为24来切割成多个小patch，单独预测结果，
    最后将预测的结果合并成原始的深度尺寸，得到最终的输出预测，然后和label计算损失
    1、需要注意的是：data文件的深度需要调整为test_cut_size的整数倍，即对不够的地方补0.比如，深度为60，需要调整成72，这样可以分割成两个深度为48的patch
    2、另外，把多个patch的预测值整合成最终的输出预测是本文件的重点。具体操作是，新建一个零值矩阵full_prob，尺寸和深度调整完成之后的data的一样，
    然后按照步距和小patch的深度将预测结果填充进去full_prob，之后，对于重叠区域（test_cut_stride<test_cut_size），需要除以重叠的次数。
    这个重叠次数的计算方法：新建一个零值矩阵full_sum,尺寸和深度调整完成之后的data的一样。然后按照步距和小patch的深度将1填充进去full_sum，
    最后得到的矩阵是这样的，没有重合的地方，值为1，有重合的地方，值就是重合的次数。最后用full_prob/full_sum就得到没有重叠的预测图像，然后按照原始深度
    从结果中取出一个矩阵，得到最终的输出预测值。
    3、这个操作有点难想到，花了两天时间理解，所以记下来。
    
"""


class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        # 读取一个data文件并归一化 、resize
        self.ct = sitk.ReadImage(data_path,sitk.sitkInt16)
        self.data_np = sitk.GetArrayFromImage(self.ct)
        self.ori_shape = self.data_np.shape
        # 在这个测试任务中，如果有label，则需要把label和原始img的尺寸调整到192*224，深度方向也会操作
        self.new_ct, self.new_seg, self.start_end_slice_list, self.crop_shape = self.resize_hw(data_path, label_path, self.n_labels)
        # self.data_np = ndimage.zoom(self.data_np, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale), order=3) # 双三次重采样
        self.resized_shape = self.new_ct.shape
        # 扩展一定数量的slices，以保证卷积下采样合理运算
        self.data_np = self.padding_img(self.new_ct, self.cut_size,self.cut_stride)
        self.padding_shape = self.data_np.shape
        # 对数据按步长进行分patch操作，以防止显存溢出
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)

        # 读取一个label文件 shape:[s,h,w]
        # self.seg = sitk.ReadImage(label_path, sitk.sitkInt8)
        # self.label_np = sitk.GetArrayFromImage(self.seg)
        # 对label文件进行和data文件一样的裁剪操作，这样在分割好之后才可以进行损失计算
        # self.label_np = ndimage.zoom(self.label_np, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale),
        #                             order=3)  # 双三次重采样
        self.label_np = self.new_seg
        if self.n_labels==2:
            self.label_np[self.label_np > 0] = 1
        self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long()

        # 预测结果保存
        self.result = None

    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])
        data = torch.FloatTensor(data).unsqueeze(0)
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_result(self):

        print('self.result.shape', self.result.shape)
        patch_s = self.result.shape[4]  # 48
        print('self.padding_shape[0]', self.padding_shape)
        N_patches_img = (self.padding_shape[2] - patch_s) // self.cut_stride + 1
        assert (self.result.shape[0] == N_patches_img)
        # print('self.ori_shape', self.ori_shape)
        full_prob = torch.zeros((self.n_labels, self.resized_shape[0], self.resized_shape[1],
                                 self.padding_shape[2]))  # itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros(
            (self.n_labels, self.resized_shape[0], self.resized_shape[1], self.padding_shape[2]))

        for s in range(N_patches_img):
            full_prob[:, :, :, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[
                s]  # 这里报错，原始尺寸的高度和宽度是512*512，results的是256*256，相加显然报错
            full_sum[:, :, :, s * self.cut_stride:s * self.cut_stride + patch_s] += 1

        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        # print(final_avg.size())
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :self.resized_shape[0], :self.resized_shape[1], :self.resized_shape[2]]
        print('img.shape', img.shape)
        # 这里将192*224*深度的预测图像返回到原始尺寸

        return img.unsqueeze(0)  # 这里好像有点问题，晚上试一下

    def normalize(self, slice, bottom=99, down=1):
        """
        normalize image with mean and std for regionnonzero,and clip the value into range
        :param slice:
        :param bottom:
        :param down:
        :return:
        """
        b = np.percentile(slice, bottom)
        t = np.percentile(slice, down)
        slice = np.clip(slice, t, b)

        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            # since the range of intensities is between 0 and 5000 ,
            # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
            # the min is replaced with -9 just to keep track of 0 intensities
            # so that we can discard those intensities afterwards when sampling random patches
            tmp[tmp == tmp.min()] = -9
            return tmp

    def resize_hw(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        print("Ori shape:",ct_array.shape, seg_array.shape)
        if classes==2:
            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1
        # # 垂直失状位这个轴RL,
        ct_RL = np.any(ct_array, axis=(0, 1))
        RL_start_slice, RL_end_slice = np.where(ct_RL)[0][[0, -1]]
        # 垂直冠状面这个轴
        ct_AP = np.any(ct_array, axis=(0, 2))
        AP_start_slice, AP_end_slice = np.where(ct_AP)[0][[0, -1]]
        # 垂直水平面这个轴
        ct_SI = np.any(ct_array, axis=(1, 2))
        SI_start_slice, SI_end_slice = np.where(ct_SI)[0][[0, -1]]
        # 提取脑部区域所在的立方体，ct和seg
        start_end_slice_list = [SI_start_slice, SI_end_slice, AP_start_slice, AP_end_slice]
        ct_array = ct_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, RL_start_slice:RL_end_slice]
        seg_array = seg_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, RL_start_slice:RL_end_slice]
        # ct_array = ct_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, :]
        # seg_array = seg_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, :]

        crop_shape = ct_array.shape
        print("Preprocessed_crop shape:", ct_array.shape, seg_array.shape)

        # 找到脑部区域所在的长方体之后，根据长度和宽度的比例和1.1667的大小，来确定填充长度或宽度的slice，考虑到小数点，
        # 填充之后的比例不一定等于1.1667
        l_w_factor = 192/224
        # 有的图像的高度、宽度也不一样，需要调整到高度和宽度是一样的，深度在下面的ndimage.zoom，以及
        if ct_array.shape[0] / ct_array.shape[1] > l_w_factor:  # 宽度方向需要填充
            tmp_ct = np.zeros((ct_array.shape[0], round(ct_array.shape[0]/l_w_factor), ct_array.shape[2]))
            delta = round((ct_array.shape[0]/l_w_factor-ct_array.shape[1])/2)
            tmp_ct[:, delta:ct_array.shape[1]+delta, :] = ct_array
            tmp_seg = np.zeros(tmp_ct.shape)
            tmp_seg[:, delta:ct_array.shape[1] + delta, :] = seg_array
        elif ct_array.shape[0] / ct_array.shape[1] < l_w_factor:  # 长度方向需要填充
            tmp_ct = np.zeros((round(ct_array.shape[1]*l_w_factor), ct_array.shape[1], ct_array.shape[2]))
            delta = round((ct_array.shape[1]*l_w_factor - ct_array.shape[0]) / 2)
            tmp_ct[delta:ct_array.shape[0] + delta, :, :] = ct_array
            tmp_seg = np.zeros(tmp_ct.shape)
            tmp_seg[delta:ct_array.shape[0] + delta, :, :] = seg_array
        else:
            tmp_ct = ct_array
            tmp_seg = seg_array
        print("Preprocessed_padding shape:", tmp_ct.shape, tmp_seg.shape)
        # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale?没看懂这个，什么叫归一化到1）,
        # ct.GetSpacing()[-1]获取深度轴的两个像素之间的间隔
        # spacing是两个像素之间的间隔（mm）,深度轴每两层之间的距离是5（每个ct扫描不一定是一样的），现在通过self.slice_down_scale=1,这个系数，
        # 通过线性插值来缩小每一层的图片的间距（假的缩小），就是增加深度层的图片个数
        # slice_depth = ct.GetSpacing()
        # 缩放到224*192
        ct_array = ndimage.zoom(tmp_ct,
                                (192/tmp_ct.shape[0], 224/tmp_ct.shape[1],
                                 ct.GetSpacing()[0] / 1.0),
                                order=3)
        seg_array = ndimage.zoom(tmp_seg,
                                 (192/tmp_ct.shape[0], 224/tmp_ct.shape[1],
                                  ct.GetSpacing()[0] / 1.0),
                                 order=0)
        print("Preprocessed_resize shape:", ct_array.shape, seg_array.shape)
        # 对数据归一化处理
        ct_array = self.normalize(ct_array)
        # # 保存为对应的格式
        # new_ct = sitk.GetImageFromArray(ct_array)
        # new_ct.SetDirection(ct.GetDirection())
        # new_ct.SetOrigin(ct.GetOrigin())
        # # new_ct.SetSpacing((1.0, ct.GetSpacing()[1] * int(1 / self.xy_down_scale), ct.GetSpacing()[2] * int(1 / self.xy_down_scale)))
        # print(new_ct.GetSpacing())
        # new_ct.SetSpacing(ct.GetSpacing())
        # new_seg = sitk.GetImageFromArray(seg_array)
        # new_seg.SetDirection(ct.GetDirection())
        # new_seg.SetOrigin(ct.GetOrigin())
        # # new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        # # new_seg.SetSpacing((self.slice_down_scale, ct.GetSpacing()[1] * int(1 / self.xy_down_scale),
        # #                    ct.GetSpacing()[2] * int(1 / self.xy_down_scale)))
        # new_seg.SetSpacing(ct.GetSpacing())
        print("Preprocessed_final shape:", ct_array.shape, seg_array.shape)
        return ct_array, seg_array, start_end_slice_list, crop_shape

    def padding_img(self, img, size, stride):
        img_h, img_w, img_s = img.shape
        leftover_s = (img_s - size) % stride

        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((img_h, img_w, s), dtype=np.float32)
        tmp_full_imgs[:, :, :img_s] = img
        print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_h, img_w, img_s = img.shape
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1

        print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, img_h, img_w, size), dtype=np.float32)

        for s in range(N_patches_img):  # loop over the full images
            patch = img[:, :, s * stride : s * stride + size]
            patches[s] = patch

        return patches  # array with all the full_imgs divided in patches


def Test_Datasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'ct/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test samples is: ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, labelpath, args=args), datapath.split('-')[-1]
