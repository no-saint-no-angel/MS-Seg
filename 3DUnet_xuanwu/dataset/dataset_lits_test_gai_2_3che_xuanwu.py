from torch._C import dtype
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import logging
import SimpleITK as sitk


class Img_DataSet(Dataset):
    def __init__(self, data_path, args):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        # 读取一个data文件并归一化 、resize
        try:
            logging.info('Preprocess on: {}'.format(data_path))
            self.new_ct, self.new_seg = self.process(data_path, self.n_labels)
        except RuntimeError:
            logging.warning('Failed on: {}'.format(data_path))
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
        # data = torch.FloatTensor(data).unsqueeze(0)
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
        patch_s = self.result.shape[2]  # 48
        print('self.padding_shape[0]', self.padding_shape)
        N_patches_img = (self.padding_shape[1] - patch_s) // self.cut_stride + 1
        assert (self.result.shape[0] == N_patches_img)
        print('self.resized_shape', self.resized_shape)
        full_prob = torch.zeros((self.n_labels, self.padding_shape[1], self.resized_shape[2], self.resized_shape[3]
                                 ))  # itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros(
            (self.n_labels, self.padding_shape[1], self.resized_shape[2], self.resized_shape[3]))

        for s in range(N_patches_img):
            full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s, :, :] += self.result[
                s]  # 这里报错，原始尺寸的高度和宽度是512*512，results的是256*256，相加显然报错
            full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s, :, :] += 1

        print('torch.min(full_sum)', torch.min(full_sum), torch.max(full_sum))
        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        # print(final_avg.size())
        print('torch.min(full_sum)', torch.min(full_prob), torch.max(full_prob))
        #assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :self.resized_shape[1], :self.resized_shape[2], :self.resized_shape[3]]
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

    def resample_image(self, itk_image, out_spacing=[1.0, 1.0, 2.0], resamplemethod=sitk.sitkNearestNeighbor):
        """
            用itk方法将原始图像resample到与目标图像一致
            :param ori_img: 原始需要对齐的itk图像
            :param target_img: 要对齐的目标itk图像
            :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
            :return:img_res_itk: 重采样好的itk图像
        """

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        # 根据输出out_spacing设置新的size
        out_size = [
            int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
            int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
            int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(itk_image)  # 需要重新采样的目标图像
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        # resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
        # 根据需要重采样图像的情况设置不同的dype
        if resamplemethod == sitk.sitkNearestNeighbor:
            resample.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
        else:
            resample.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32

        # resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resample.SetInterpolator(resamplemethod)
        # resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)

    def process(self, data_path, classes=None):
        anli_name = os.path.basename(data_path)
        ct_t2_path = os.path.join(data_path, anli_name+'_t2.nii.gz')
        ct_t1_path = os.path.join(data_path, anli_name+'_t1.nii.gz')
        ct_flair_path = os.path.join(data_path, anli_name+'_t2flair.nii.gz')
        seg_path = os.path.join(data_path, anli_name+'.nii.gz')
        ct_t2 = sitk.ReadImage(ct_t2_path, sitk.sitkInt16)
        ct_t1 = sitk.ReadImage(ct_t1_path, sitk.sitkInt16)
        ct_flair = sitk.ReadImage(ct_flair_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)

        # 重采样
        resample_spacing = [0.4688, 0.4688, 1.55]
        ct_t2 = self.resample_image(ct_t2, out_spacing=resample_spacing, resamplemethod=sitk.sitkLinear)
        ct_t1_af_r = self.resample_image(ct_t1, out_spacing=resample_spacing, resamplemethod=sitk.sitkLinear)
        ct_flair = self.resample_image(ct_flair, out_spacing=resample_spacing, resamplemethod=sitk.sitkLinear)
        seg = self.resample_image(seg, out_spacing=resample_spacing, resamplemethod=sitk.sitkNearestNeighbor)

        print('Spacing_and_dimension_bf_resample', ct_t1_af_r.GetSpacing(), ct_t1_af_r.GetWidth(),
              ct_t1_af_r.GetHeight(),
              ct_t1_af_r.GetDepth())

        ct_t2_array = sitk.GetArrayFromImage(ct_t2)
        ct_t1_array = sitk.GetArrayFromImage(ct_t1_af_r)
        ct_flair_array = sitk.GetArrayFromImage(ct_flair)
        seg_array = sitk.GetArrayFromImage(seg)
        print("Ori shape:", ct_t2_array.shape, seg_array.shape)
        if classes == 2:
            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1

        # 颅脑bunding box
        # 垂直失状位这个轴RL（x）
        ct_RL = np.any(ct_t2_array, axis=(0, 1))
        RL_start_slice, RL_end_slice = np.where(ct_RL)[0][[0, -1]]
        # 垂直冠状面这个轴AP(y)
        ct_AP = np.any(ct_t2_array, axis=(0, 2))
        AP_start_slice, AP_end_slice = np.where(ct_AP)[0][[0, -1]]
        # # 垂直水平面这个轴SI(z)
        ct_SI = np.any(ct_t2_array, axis=(1, 2))
        SI_start_slice, SI_end_slice = np.where(ct_SI)[0][[0, -1]]
        # 提取
        ct_t2_array = ct_t2_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, RL_start_slice:RL_end_slice]
        ct_t1_array = ct_t1_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, RL_start_slice:RL_end_slice]
        ct_flair_array = ct_flair_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice,
                         RL_start_slice:RL_end_slice]
        seg_array = seg_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, RL_start_slice:RL_end_slice]
        print("Preprocessed_crop shape:", ct_t2_array.shape, seg_array.shape)
        # # 这里的shape的顺序是zyx（SI-AP-RL），在itk-snap的切片中的顺序是xyz（RL-AP-SI），刚好反了一下。

        # Fill and crop到指定大小
        SI_2_AP = [320, 320]
        delta_0 = round(abs(ct_t2_array.shape[1] - SI_2_AP[0]) / 2)
        delta_1 = round(abs(ct_t2_array.shape[2] - SI_2_AP[1]) / 2)
        if delta_0 + delta_1 != 0:
            if ct_t2_array.shape[1] >= SI_2_AP[0] and ct_t2_array.shape[2] >= SI_2_AP[1]:  # 宽度,高度方向需要裁剪
                tmp_ct_t2 = ct_t2_array[:, delta_0:SI_2_AP[0] + delta_0, delta_1:SI_2_AP[1] + delta_1]
                tmp_ct_t1 = ct_t1_array[:, delta_0:SI_2_AP[0] + delta_0, delta_1:SI_2_AP[1] + delta_1]
                tmp_ct_flair = ct_flair_array[:, delta_0:SI_2_AP[0] + delta_0, delta_1:SI_2_AP[1] + delta_1]
                tmp_seg = seg_array[:, delta_0:SI_2_AP[0] + delta_0, delta_1:SI_2_AP[1] + delta_1]
            elif ct_t2_array.shape[1] <= SI_2_AP[0] and ct_t2_array.shape[2] <= SI_2_AP[1]:  # 宽度,高度方向需要填充
                tmp_ct_t2 = np.zeros((ct_t2_array.shape[0], SI_2_AP[0], SI_2_AP[1]))
                tmp_ct_t1 = np.zeros(tmp_ct_t2.shape)
                tmp_ct_flair = np.zeros(tmp_ct_t2.shape)
                tmp_seg = np.zeros(tmp_ct_t2.shape)

                tmp_ct_t2[:, delta_0:ct_t2_array.shape[1] + delta_0,
                delta_1:ct_t2_array.shape[2] + delta_1] = ct_t2_array
                tmp_ct_t1[:, delta_0:ct_t2_array.shape[1] + delta_0,
                delta_1:ct_t2_array.shape[2] + delta_1] = ct_t1_array
                tmp_ct_flair[:, delta_0:ct_t2_array.shape[1] + delta_0,
                delta_1:ct_t2_array.shape[2] + delta_1] = ct_flair_array
                tmp_seg[:, delta_0:ct_t2_array.shape[1] + delta_0, delta_1:ct_t2_array.shape[2] + delta_1] = seg_array

            elif ct_t2_array.shape[1] >= SI_2_AP[0] and ct_t2_array.shape[2] < SI_2_AP[1]:  # 宽度需要裁剪，高度需要填充
                # 宽度裁剪
                tmp_ct_t2_1 = ct_t2_array[:, delta_0:SI_2_AP[0] + delta_0, :]
                tmp_ct_t1_1 = ct_t1_array[:, delta_0:SI_2_AP[0] + delta_0, :]
                tmp_ct_flair_1 = ct_flair_array[:, delta_0:SI_2_AP[0] + delta_0, :]
                tmp_seg_1 = seg_array[:, delta_0:SI_2_AP[0] + delta_0, :]

                # 高度填充
                tmp_ct_t2 = np.zeros((ct_t2_array.shape[0], SI_2_AP[0], SI_2_AP[1]))
                tmp_ct_t1 = np.zeros(tmp_ct_t2.shape)
                tmp_ct_flair = np.zeros(tmp_ct_t2.shape)
                tmp_seg = np.zeros(tmp_ct_t2.shape)

                tmp_ct_t2[:, :, delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_ct_t2_1
                tmp_ct_t1[:, :, delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_ct_t1_1
                tmp_ct_flair[:, :, delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_ct_flair_1
                tmp_seg[:, :, delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_seg_1

            else:  # ct_t2_array.shape[1] >= SI_2_AP[0] and ct_t2_array.shape[2] >= SI_2_AP[1]:  # 宽度需要填充，高度需要裁剪
                # 高度裁剪
                tmp_ct_t2_1 = ct_t2_array[:, :, delta_1:SI_2_AP[1] + delta_1]
                tmp_ct_t1_1 = ct_t1_array[:, :, delta_1:SI_2_AP[1] + delta_1]
                tmp_ct_flair_1 = ct_flair_array[:, :, delta_1:SI_2_AP[1] + delta_1]
                tmp_seg_1 = seg_array[:, :, delta_1:SI_2_AP[1] + delta_1]
                # 宽度填充
                tmp_ct_t2 = np.zeros((ct_t2_array.shape[0], SI_2_AP[0], SI_2_AP[1]))
                tmp_ct_t1 = np.zeros(tmp_ct_t2.shape)
                tmp_ct_flair = np.zeros(tmp_ct_t2.shape)
                tmp_seg = np.zeros(tmp_ct_t2.shape)

                tmp_ct_t2[:, delta_0:tmp_ct_t2_1.shape[1] + delta_0, :] = tmp_ct_t2_1
                tmp_ct_t1[:, delta_0:tmp_ct_t2_1.shape[1] + delta_0, :] = tmp_ct_t1_1
                tmp_ct_flair[:, delta_0:tmp_ct_t2_1.shape[1] + delta_0, :] = tmp_ct_flair_1
                tmp_seg[:, delta_0:tmp_ct_t2_1.shape[1] + delta_0, :] = tmp_seg_1
        else:
            tmp_ct_t2 = ct_t2_array
            tmp_ct_t1 = ct_t1_array
            tmp_ct_flair = ct_flair_array
            tmp_seg = seg_array
        print("Preprocessed_padding shape:", tmp_ct_t2.shape, tmp_seg.shape)
        print('')

        # Z-score
        ct_t2_array = self.normalize(tmp_ct_t2)
        ct_t1_array = self.normalize(tmp_ct_t1)
        ct_flair_array = self.normalize(tmp_ct_flair)

        print("Preprocessed_final shape:", ct_t2_array.shape, tmp_seg.shape)
        # 合并三个模态
        assert ct_t2_array.shape == ct_t1_array.shape and ct_t2_array.shape == ct_flair_array.shape
        merged_mri = np.zeros([3, ct_t2_array.shape[0], ct_t2_array.shape[1], ct_t2_array.shape[2]])
        merged_mri[0, :] = ct_t2_array
        merged_mri[1, :] = ct_t1_array
        merged_mri[2, :] = ct_flair_array
        return merged_mri, tmp_seg

    def padding_img(self, img, size, stride):
        img_cha, img_s, img_h, img_w = img.shape
        leftover_s = (img_s - size) % stride

        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((img_cha, s, img_h, img_w), dtype=np.float32)
        tmp_full_imgs[:, :img_s, :, :] = img
        print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_cha, img_s, img_h, img_w = img.shape
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1

        print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, img_cha, size, img_h, img_w), dtype=np.float32)

        for s in range(N_patches_img):  # loop over the full images
            patch = img[:, s * stride: s * stride + size, :, :]
            patches[s] = patch

        return patches  # array with all the full_imgs divided in patches


def Test_Datasets(dataset_path, args):
    # data_list = sorted(glob(os.path.join(dataset_path, 'ct/*')))
    data_list = os.listdir(dataset_path)
    # label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test_3cha samples is: ", len(data_list))
    for anli_dir in data_list:
        datapath = os.path.join(dataset_path, anli_dir)
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, args=args), anli_dir.split('-')[-1]
