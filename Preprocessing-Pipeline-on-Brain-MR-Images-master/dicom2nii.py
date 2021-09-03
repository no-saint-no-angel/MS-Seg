import SimpleITK as sitk
import numpy as np
import os
from os.path import join
import shutil
""" 
    这个文件的功能是，将dicom文件转换成.nii格式，
    其中dicom文件是个单独的文件夹，包含MRI所有序列的图像，从命名不能区分单张dicom属于哪个模态
    输出形式是，每个案例一个文件夹，里面包含T2,T1,T2FLAIR三个模态
    具体实现：
    sitk有读取单个序列的功能，所以先把每个模态的dicom放置到单独的文件夹（按照每个模态dicom的个数来区分），转换到.nii格式
    然后保存到输出文件夹。（label先不管，等预处理完了之后在把label和三个模态的.nii格式统一放到案例文件夹下）
"""


def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, os.path.join(dstpath, fname))          # 复制文件
        print ("copy %s -> %s"%(srcfile, os.path.join(dstpath, fname)))


def dicomseries2nii(case_path, nii_save_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    origin = image.GetOrigin()  # x, y, z
    spacing = image.GetSpacing()  # x, y, z
    image_nii = sitk.GetImageFromArray(image_array)
    image_nii.SetOrigin(image.GetOrigin())
    image_nii.SetDirection(image.GetDirection())
    image_nii.SetSpacing(image.GetSpacing())
    sitk.WriteImage(image_nii, nii_save_path)

# 输入文件夹
input_dir = 'D:\\projects\\lianren\\data_xuanwu\\anli_dir'
# case_path = 'D:\\projects\\lianren\\data_xuanwu\\msai_00002_01\\msai_00002_01_40_60'
# nii_save_path = './data_output/msai_00002_01_40_60.nii'
# 输出文件夹
out_dir = './output'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# 存放三个模态dicom的临时文件夹
T1_tmp_dir = './tmp/T1_tmp_dir'
T2_tmp_dir = './tmp/T2_tmp_dir'
T2FLAIR_tmp_dir = './tmp/T2FLAIR_tmp_dir'
# 提取三个模态的数据到临时文件夹
anli_list = os.listdir(input_dir)
Numbers = len(anli_list)
print('Total numbers of samples is :', Numbers)
for anli_dir in anli_list:
    # anli_dir_path = os.path.join(input_dir, anli_dir)
    # 计数，根据count大小来将某个dicom文件复制到指定的临时文件夹
    count = 0
    anli_dir_path = os.path.join(os.path.join(input_dir, anli_dir), anli_dir)
    anli_dicom_list = os.listdir(anli_dir_path)
    for anli_dicom in anli_dicom_list:
        if anli_dicom.endswith('.dcm'):
            count += 1
            anli_dicom_path = os.path.join(anli_dir_path, anli_dicom)
            if count < 21:  # T2
                mycopyfile(anli_dicom_path, os.path.join(T2_tmp_dir, anli_dir))
            elif count < 41:  # T1
                mycopyfile(anli_dicom_path, os.path.join(T1_tmp_dir, anli_dir))
            elif count < 61:  # T2FLAIR
                mycopyfile(anli_dicom_path, os.path.join(T2FLAIR_tmp_dir, anli_dir))
            else:
                break

# dicom --> .nii
anli_list_t2 = os.listdir(T2_tmp_dir)
for anli_t2 in anli_list_t2:
    save_path = os.path.join(out_dir, anli_t2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    t2_name = anli_t2+'_t2.nii'
    t1_name = anli_t2 + '_t1.nii'
    t2flair_name = anli_t2 + '_t2flair.nii'
    # T2
    anli_t2_path = os.path.join(T2_tmp_dir, anli_t2)
    print('processing the t2 series'+'*'*20)
    dicomseries2nii(anli_t2_path, os.path.join(save_path, t2_name))
    # T1
    anli_t1_path = os.path.join(T1_tmp_dir, anli_t2)
    print('processing the t1 series' + '*' * 20)
    dicomseries2nii(anli_t1_path, os.path.join(save_path, t1_name))
    # T2FLAIR
    anli_t2flair_path = os.path.join(T2FLAIR_tmp_dir, anli_t2)
    print('processing the t2flair series' + '*' * 20)
    dicomseries2nii(anli_t2flair_path, os.path.join(save_path, t2flair_name))



# # 读取单张dicom
# single_slice_path = 'D:\\projects\\lianren\\data_xuanwu\\msai_00002_01\\msai_00002_01\\.DS_Store'
# single_slice_nii_save_path = './data_output/msai_00002_01_001afeb5.nii'
# image = sitk.ReadImage(single_slice_path)
# image_array = sitk.GetArrayFromImage(image) # z, y, x
# origin = image.GetOrigin()  # x, y, z
# spacing = image.GetSpacing()  # x, y, z
# image_nii = sitk.GetImageFromArray(image_array)
# image_nii.SetOrigin(image.GetOrigin())
# image_nii.SetDirection(image.GetDirection())
# image_nii.SetSpacing(image.GetSpacing())
# sitk.WriteImage(image_nii, single_slice_nii_save_path)

