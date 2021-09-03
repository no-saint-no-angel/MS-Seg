import numpy as np
import os
import SimpleITK as sitk
import random
import logging
from os.path import join
import config


class LITS_preprocess:
    def __init__(self, raw_data_ct_path, raw_data_mask_path,fixed_dataset_path, args):
        self.raw_ct_root_path = raw_data_ct_path
        self.raw_label_root_path = raw_data_mask_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels  # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
        # self.upper = args.upper
        # self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        # self.xy_down_scale = args.xy_down_scale
        # self.slice_down_scale = args.slice_down_scale

        self.valid_rate = args.valid_rate

    def fix_data(self):
        if not os.path.exists(self.fixed_path):    # 创建保存目录
            os.makedirs(join(self.fixed_path,'mri'))
            os.makedirs(join(self.fixed_path, 't2'))
            os.makedirs(join(self.fixed_path, 'label'))
        # 在self.raw_ct_root_path文件夹下，进入每个案例的子文件夹，找到T1,T2,FLAIR三个.gz文件，重新命名（依据案例名字）
        # self.raw_label_root_path文件夹下，找到和ct案例相同文件名的子文件夹，找到ManualSegmentation_1.nii.gz文件，重命名（依据案例名字）
        ct_file_list = os.listdir(self.raw_ct_root_path)
        Numbers = len(ct_file_list)
        print('Total numbers of samples is :',Numbers)
        for i in range(Numbers):
            anli_name = ct_file_list[i]
            anli_dir = os.path.join(self.raw_ct_root_path, anli_name)

            t2_ct_file_path = os.path.join(anli_dir, anli_name+'_t2.nii.gz')
            t1_ct_file_path = os.path.join(anli_dir, anli_name+'_t1.nii.gz')
            flair_ct_file_path = os.path.join(anli_dir, anli_name+'_t2flair.nii.gz')
            label_dir = os.path.join(self.raw_label_root_path, anli_name)
            label_file_path = os.path.join(label_dir, anli_name+'.nii.gz')
            print("==== {} | {}/{} ====".format(label_file_path, i+1,Numbers))
            # ct和label不在同一个文件夹下
            try:
                logging.info('Preprocess on: {}'.format(t2_ct_file_path))
                new_ct_t2, new_ct_t1, new_ct_flair, new_seg = self.process(t2_ct_file_path, t1_ct_file_path,
                                                                           flair_ct_file_path, label_file_path,
                                                                           classes=self.classes)
                # # 融合三个通道的数据
                merged_mri = self.merge_3che(new_ct_t2, new_ct_t1, new_ct_flair)

                if merged_mri.any() != None and new_seg != None:
                    merged_mri_name = 'volume_3cha_' + str(anli_name) + '.npy'
                    seg_mri_name = 'segmentation_' + str(anli_name) + '.nii.gz'
                    new_t2_name = str(anli_name) + '_t2.nii.gz'
                    np.save(os.path.join(os.path.join(self.fixed_path, 'mri'), merged_mri_name), merged_mri)
                    sitk.WriteImage(new_ct_t2, os.path.join(os.path.join(self.fixed_path, 't2'), new_t2_name))
                    sitk.WriteImage(new_seg, os.path.join(os.path.join(self.fixed_path, 'label'), seg_mri_name))

            except RuntimeError:
                logging.warning('Failed on: {}'.format(t2_ct_file_path))


    def merge_3che(self, t2, t1, flair):
        t2_array = sitk.GetArrayFromImage(t2)
        t1_array = sitk.GetArrayFromImage(t1)
        flair_array = sitk.GetArrayFromImage(flair)
        # # 不相等的情况只会出现在第三个轴，
        # if t2_array.shape != t1_array.shape or t2_array.shape != flair_array.shape:
        #     shape_list = [t2_array.shape[2], t1_array.shape[2], flair_array.shape[2]]
        #     max_shape = shape_list[np.argmax(shape_list)]
        #     if t2_array.shape[2] != max_shape:
        #         t2_array_tmp = np.zeros([t2_array.shape[0], t2_array.shape[1], max_shape])
        #         t2_array_tmp[:, :, :t2_array.shape[2]] = t2_array
        #         t2_array = t2_array_tmp
        #     if t1_array.shape[2] != max_shape:
        #         t1_array_tmp = np.zeros([t1_array.shape[0], t1_array.shape[1], max_shape])
        #         t1_array_tmp[:, :, :t1_array.shape[2]] = t1_array
        #         t1_array = t1_array_tmp
        #     if flair_array.shape[2] != max_shape:
        #         flair_array_tmp = np.zeros([flair_array.shape[0], flair_array.shape[1], max_shape])
        #         flair_array_tmp[:, :, :flair_array.shape[2]] = flair_array
        #         flair_array = flair_array_tmp

        assert t2_array.shape == t1_array.shape and t2_array.shape == flair_array.shape
        merged_mri = np.zeros([3, t2_array.shape[0], t2_array.shape[1], t2_array.shape[2]])
        merged_mri[0, :] = t2_array
        merged_mri[1, :] = t1_array
        merged_mri[2, :] = flair_array

        return merged_mri

        # 对医疗图像进行重采样，仅仅需要将out_spacing替换成自己想要的输出即可
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

    def process(self, ct_t2_path, ct_t1_path, ct_flair_path, seg_path, classes=None):
        ct_t2 = sitk.ReadImage(ct_t2_path, sitk.sitkInt16)
        ct_t1 = sitk.ReadImage(ct_t1_path, sitk.sitkInt16)
        ct_flair = sitk.ReadImage(ct_flair_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        print('Spacing_and_dimension_bf_resample', ct_t2.GetSpacing(), ct_t2.GetWidth(), ct_t2.GetHeight(),
              ct_t2.GetDepth())
        # 重采样
        resample_spacing = [0.4688, 0.4688, 1.55]
        ct_t2 = self.resample_image(ct_t2, out_spacing=resample_spacing, resamplemethod=sitk.sitkLinear)
        ct_t1_af_r = self.resample_image(ct_t1, out_spacing=resample_spacing, resamplemethod=sitk.sitkLinear)
        ct_flair = self.resample_image(ct_flair, out_spacing=resample_spacing, resamplemethod=sitk.sitkLinear)
        seg = self.resample_image(seg, out_spacing=resample_spacing, resamplemethod=sitk.sitkNearestNeighbor)

        print('Spacing_and_dimension_bf_resample', ct_t1_af_r.GetSpacing(), ct_t1_af_r.GetWidth(), ct_t1_af_r.GetHeight(),
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
        ct_flair_array = ct_flair_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, RL_start_slice:RL_end_slice]
        seg_array = seg_array[SI_start_slice:SI_end_slice, AP_start_slice:AP_end_slice, RL_start_slice:RL_end_slice]
        print("Preprocessed_crop shape:", ct_t2_array.shape, seg_array.shape)
        # # 这里的shape的顺序是zyx（SI-AP-RL），在itk-snap的切片中的顺序是xyz（RL-AP-SI），刚好反了一下。

        # Fill and crop到指定大小
        SI_2_AP = [320, 320]
        delta_0 = round(abs(ct_t2_array.shape[1] - SI_2_AP[0]) / 2)
        delta_1 = round(abs(ct_t2_array.shape[2] - SI_2_AP[1]) / 2)
        if delta_0+delta_1 != 0:
            if ct_t2_array.shape[1] >= SI_2_AP[0] and ct_t2_array.shape[2] >= SI_2_AP[1]:  # 宽度,高度方向需要裁剪
                tmp_ct_t2 = ct_t2_array[:, delta_0:SI_2_AP[0]+delta_0, delta_1:SI_2_AP[1]+delta_1]
                tmp_ct_t1 = ct_t1_array[:, delta_0:SI_2_AP[0]+delta_0, delta_1:SI_2_AP[1]+delta_1]
                tmp_ct_flair = ct_flair_array[:, delta_0:SI_2_AP[0]+delta_0, delta_1:SI_2_AP[1]+delta_1]
                tmp_seg = seg_array[:, delta_0:SI_2_AP[0]+delta_0, delta_1:SI_2_AP[1]+delta_1]
            elif ct_t2_array.shape[1] <= SI_2_AP[0] and ct_t2_array.shape[2] <= SI_2_AP[1]:  # 宽度,高度方向需要填充
                tmp_ct_t2 = np.zeros((ct_t2_array.shape[0], SI_2_AP[0], SI_2_AP[1]))
                tmp_ct_t1 = np.zeros(tmp_ct_t2.shape)
                tmp_ct_flair = np.zeros(tmp_ct_t2.shape)
                tmp_seg = np.zeros(tmp_ct_t2.shape)

                tmp_ct_t2[:, delta_0:ct_t2_array.shape[1] + delta_0, delta_1:ct_t2_array.shape[2] + delta_1] = ct_t2_array
                tmp_ct_t1[:, delta_0:ct_t2_array.shape[1] + delta_0, delta_1:ct_t2_array.shape[2] + delta_1] = ct_t1_array
                tmp_ct_flair[:, delta_0:ct_t2_array.shape[1] + delta_0, delta_1:ct_t2_array.shape[2] + delta_1] = ct_flair_array
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

                tmp_ct_t2[:, :,delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_ct_t2_1
                tmp_ct_t1[:, :,delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_ct_t1_1
                tmp_ct_flair[:, :,delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_ct_flair_1
                tmp_seg[:, :,delta_1:tmp_ct_t2_1.shape[2] + delta_1] = tmp_seg_1

            else:  #  ct_t2_array.shape[1] >= SI_2_AP[0] and ct_t2_array.shape[2] >= SI_2_AP[1]:  # 宽度需要填充，高度需要裁剪
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

        # # 找到病灶开始和结束的slice（垂直水平面方向），并各向外扩张（这个是对训练的数据进行的操作，根据分割mask找到前景区域，只训练前景区域，
        # z = np.any(seg_array, axis=(1, 2))
        # start_slice, end_slice = np.where(z)[0][[0, -1]]
        # # 两个方向上各扩张个slice
        # if start_slice - self.expand_slice < 0:
        #     start_slice = 0
        # else:
        #     start_slice -= self.expand_slice
        #
        # if end_slice + self.expand_slice >= seg_array.shape[0]:
        #     end_slice = seg_array.shape[0] - 1
        # else:
        #     end_slice += self.expand_slice
        #
        # print("Cut out range:", str(start_slice) + '--' + str(end_slice))
        # # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
        # if end_slice - start_slice + 1 < self.size:
        #     print('Too little slice，give up the sample:', ct_t2_path)
        #     return None, None
        # ct_t2_array = ct_t2_array[start_slice:end_slice + 1, :, :]
        # ct_t1_array = ct_t1_array[start_slice:end_slice + 1, :, :]
        # ct_flair_array = ct_flair_array[start_slice:end_slice + 1, :, :]
        # seg_array = tmp_seg[start_slice:end_slice + 1, :, :]
        # print("Preprocessed_expand shape:", ct_t2_array.shape, seg_array.shape)

        # 保存
        # t2
        new_ct_t2 = sitk.GetImageFromArray(ct_t2_array)
        new_ct_t2.SetDirection(ct_t2.GetDirection())
        new_ct_t2.SetOrigin(ct_t2.GetOrigin())
        new_ct_t2.SetSpacing(ct_t2.GetSpacing())
        # t1
        new_ct_t1 = sitk.GetImageFromArray(ct_t1_array)
        new_ct_t1.SetDirection(ct_t2.GetDirection())
        new_ct_t1.SetOrigin(ct_t2.GetOrigin())
        new_ct_t1.SetSpacing(ct_t2.GetSpacing())
        # flair
        new_ct_flair = sitk.GetImageFromArray(ct_flair_array)
        new_ct_flair.SetDirection(ct_t2.GetDirection())
        new_ct_flair.SetOrigin(ct_t2.GetOrigin())
        new_ct_flair.SetSpacing(ct_t2.GetSpacing())
        # seg
        new_seg = sitk.GetImageFromArray(tmp_seg)
        new_seg.SetDirection(ct_t2.GetDirection())
        new_seg.SetOrigin(ct_t2.GetOrigin())
        new_seg.SetSpacing(ct_t2.GetSpacing())
        print("Preprocessed_final shape:", ct_t2_array.shape, seg_array.shape)
        return new_ct_t2, new_ct_t1, new_ct_flair, new_seg

    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path, "mri"))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num*(1-self.valid_rate))]
        val_name_list = data_name_list[int(data_num*(1-self.valid_rate)):int(data_num*((1-self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'mri', name)
            seg_path = os.path.join(self.fixed_path, 'label', name.replace('volume_3cha', 'segmentation').replace('.npy', '.nii'))
            # seg_path = os.path.join(self.fixed_path, 'label', name)
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()

if __name__ == '__main__':
    #raw_data_ct_path = "D:\\projects\\deeplearning\\3D\\Preprocessing-Pipeline-on-Brain-MR-Images-master\\data\\anli_prep_reg_sk_N4_eh"
    #raw_data_mask_path = "D:\\projects\\deeplearning\\3D\\Preprocessing-Pipeline-on-Brain-MR-Images-master\\data\\data_label"
    raw_data_ct_path = "/home/zhenghuadong/project/Preprocessing-Pipeline-on-Brain-MR-Images-master/data/anli_prep_reg_sk_N4_eh/"
    raw_data_mask_path = "/data/luoyonggui/ZXL/"

    fixed_dataset_path = './fixed_data_3cha_for_test'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = config.args
    tool = LITS_preprocess(raw_data_ct_path, raw_data_mask_path,fixed_dataset_path, args)
    tool.fix_data()  # 对原始图像进行修剪并保存
    tool.write_train_val_name_list()  # 创建索引txt文件