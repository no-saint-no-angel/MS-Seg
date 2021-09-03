import numpy as np
import nibabel as nib
import os
import pandas as pd
import SimpleITK as sitk
import shutil
import matplotlib.pyplot as plt
#import dicom2nifti
import glob


def mycopyfile(srcfile,dstpath):
    if not os.path.isdir(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        current_list = glob.glob(os.path.join(srcfile, '*'))
        for x in current_list:
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)                       # 创建路径
            shutil.copy2(x, dstpath)          # 复制文件
        print("copy %s -> %s"%(srcfile, dstpath))


def dcm2nii_sitk(path_read, path_save):
    # 获取案例名字
    anli_dir_name = os.path.basename(path_save)
    # 保存dicom系列,不同机器的三个序列，T1,T2FLAIR,T2
    dicom_series_t1 = ['T1FSPGROAx', 't1_fl2d_tra', 'T1W_FE', 'T1W_FFE']
    dicom_series_t2 = ['T2PROPELLEROAx', 't2_tse_tra_p2', 't2_tse_tra', 'T2traMV', 'T2W_TSE']
    dicom_series_t2flair = ['T2FLAIROAxfs', 't2_tirm_tra_dark-fluid', 't2_tirm_tra_dark-fluid_p2', 'FLAIR_MV', 'FLAIR_T2']
    dicom_series = dicom_series_t1 + dicom_series_t2 + dicom_series_t2flair
    # dicom_series = ['t1_fl2d_tra', 't2_tirm_tra_dark-fluid', 't2_tse_tra_p2']
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    file_reader = sitk.ImageFileReader()
    N = len(seriesIDs)
    lens = np.zeros([N])
    # 保存每个series的个数以及满足需要序列的总个数
    count_saved_num_big = 0
    count_saved_num_t1 = 0
    count_saved_num_t2 = 0
    count_saved_num_t2flair = 0
    saved_true_false = 0
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        # 读取单张dicom获取描述符
        file_reader.SetFileName(dicom_names[0])
        file_reader.ReadImageInformation()
        series_description = file_reader.GetMetaData(
            "0008|103e")  # 序列的描述符，为CT WB 5.0 B31f、PET WB (AC)和ThorRoutine 2.0 B40f
        if series_description.replace(" ", "") in dicom_series:
            lens[i] = len(dicom_names)
            print('Serie ' + series_description +' have '+str(len(dicom_names))+' slices.')
            # 获取整个序列的dicom
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            # 保存
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            if series_description.replace(" ", "") in dicom_series_t1:
                nii_name = anli_dir_name + '_t1.nii.gz'
                count_saved_num_t1 += 1
                if count_saved_num_t1 == 1:
                    # nii_name = anli_dir_name + '_t1_'+str(count_saved_num_t1-1)+'.nii.gz'
                    sitk.WriteImage(image, os.path.join(path_save, nii_name))

            if series_description.replace(" ", "") in dicom_series_t2flair:
                nii_name = anli_dir_name + '_t2flair.nii.gz'
                count_saved_num_t2flair += 1
                if count_saved_num_t2flair == 1:
                    # nii_name = anli_dir_name + '_t2flair_' + str(count_saved_num_t2flair - 1) + '.nii.gz'
                    sitk.WriteImage(image, os.path.join(path_save, nii_name))
            if series_description.replace(" ", "") in dicom_series_t2:
                nii_name = anli_dir_name + '_t2.nii.gz'
                count_saved_num_t2 += 1
                if count_saved_num_t2 == 1:
                    # nii_name = anli_dir_name + '_t2_' + str(count_saved_num_t2 - 1) + '.nii.gz'
                    sitk.WriteImage(image, os.path.join(path_save, nii_name))


            # sitk.WriteImage(image, os.path.join(path_save, nii_name))
    count_saved_num_big = count_saved_num_t2 + count_saved_num_t1 + count_saved_num_t2flair
    if count_saved_num_big == 3:
        print(str(anli_dir_name) + ' have three series.')
        saved_true_false = 1
    else:
        print(str(anli_dir_name) + ' have no three series.'+'&'*20)

    return saved_true_false, anli_dir_name, count_saved_num_big


input_dir = '/data/luoyonggui/ZXL'
# 保存所有的案例，三个模态的或者其他
output_dir = './data_dicom2nii/output_only3/output/'
output_dir_4 = './data_dicom2nii/output_only3/output_4/'
output_dir_2 = './data_dicom2nii/output_only3/output_2/'
output_dir_others = './data_dicom2nii/output_only3/output_others/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir_4):
    os.makedirs(output_dir_4)
if not os.path.exists(output_dir_2):
    os.makedirs(output_dir_2)
if not os.path.exists(output_dir_others):
    os.makedirs(output_dir_others)
anli_list = os.listdir(input_dir)
Numbers = len(anli_list)
print('Total numbers of samples is :', Numbers)
count = 0
num_transfered = 0
un_transfered_dir_name_list_others = []
# 保存有四个模态的anli
un_transfered_dir_name_list_4 = []
# 保存有两个模态的anli
un_transfered_dir_name_list_2 = []
for anli_dir in anli_list:
    count += 1
    print('processing the '+str(count)+' dir:'+str(anli_dir)+'.'*20)
    # 在自己电脑上的路径
    # anli_dir_path = os.path.join(os.path.join(input_dir, anli_dir), anli_dir)
    # 在服务器上的路径
    anli_dir_path = os.path.join(input_dir, anli_dir)
    output_dir_anli = os.path.join(output_dir, anli_dir)
    saved_true_false, anli_dir_name, count_saved_num = dcm2nii_sitk(anli_dir_path, output_dir_anli)
    if saved_true_false == 0:
        if count_saved_num == 4:
            un_transfered_dir_name_list_4.append(anli_dir_name)
            mycopyfile(output_dir_anli, os.path.join(output_dir_4, anli_dir))
        elif count_saved_num == 2:
            un_transfered_dir_name_list_2.append(anli_dir_name)
            mycopyfile(output_dir_anli, os.path.join(output_dir_2, anli_dir))
        else:
            un_transfered_dir_name_list_others.append([anli_dir_name, count_saved_num])
            mycopyfile(output_dir_anli, os.path.join(output_dir_others, anli_dir))
    num_transfered += saved_true_false
    print('Conversion progress:'+str(num_transfered)+'/'+str(count))
print('num_of_2series:', str(len(un_transfered_dir_name_list_2)), ' un_transfered_dir_name_list_2:', un_transfered_dir_name_list_2)
print('num_of_4series:', str(len(un_transfered_dir_name_list_4)), ' un_transfered_dir_name_list_4:', un_transfered_dir_name_list_4)
print('num_of_others_series:', str(len(un_transfered_dir_name_list_others)), ' un_transfered_dir_name_list_others:', un_transfered_dir_name_list_others)

