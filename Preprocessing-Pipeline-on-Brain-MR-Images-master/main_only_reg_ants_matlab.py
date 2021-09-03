import shutil
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import os
import logging
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from scipy.signal import medfilt
from sklearn.cluster import KMeans
import ants
import numpy as np
import SimpleITK as sitk



def createDir(path):
    _path = Path(path)
    if not _path.is_dir():
        _path.mkdir()
    return


def runACPCDetect(niifile, acpcDetectPath='./utils/acpcdetect_V2.0_macOS10.12.6/bin/acpcdetect'):
    command = [acpcDetectPath, "-no-tilt-correction", "-center-AC", "-nopng", "-noppm", "-i", niifile]
    subprocess.call(command, stdout=open(os.devnull, "r"), stderr=subprocess.STDOUT)
    return


def orient2std(src_path, dst_path):
    command = ["fslreorient2std", src_path, dst_path]
    subprocess.call(command)
    return


def registration_ants_f(m_img_path, m_seg_path, save_path, f_img_path):
    '''
    ants.registration()函数的返回值是一个字典：
        warpedmovout: 配准到fixed图像后的moving图像
        warpedfixout: 配准到moving图像后的fixed图像
        fwdtransforms: 从moving到fixed的形变场
        invtransforms: 从fixed到moving的形变场

    type_of_transform参数的取值可以为：
        Rigid：刚体
        Affine：仿射配准，即刚体+缩放
        ElasticSyN：仿射配准+可变形配准，以MI为优化准则，以elastic为正则项
        SyN：仿射配准+可变形配准，以MI为优化准则
        SyNCC：仿射配准+可变形配准，以CC为优化准则
    '''
    f_img = ants.image_read(f_img_path)
    m_img = ants.image_read(m_img_path)
    # m_label = ants.image_read(m_seg_path)
    # 图像配准
    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='Rigid')
    #print('3', mytx['fwdtransforms'])
    # 将形变场作用于moving图像，得到配准后的图像，interpolator也可以选择"nearestNeighbor"等
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                       interpolator="nearestNeighbor")

    # 将配准后图像的direction/origin/spacing和原图保持一致
    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)
    # 保存
    ants.image_write(warped_img, save_path)

    # 判断正在进行配准的图像（m_img）是否是t1序列的，如果是，则把m_seg_path下的所有mask都进行配准操作
    m_img_name = str(m_img_path).split("/")[-1]
    # a = str(m_img_name).split("_")[-1]
    if str(m_img_name).split("_")[-1] == 't1.nii.gz':
        niiGzPaths = Path(m_seg_path).glob('**/*.gz')
        niiGzFiles = [niiGzPath for niiGzPath in niiGzPaths if niiGzPath.is_file()]
        for m_label_file in niiGzFiles:
            # print('')
            m_label_path = m_label_file.as_posix()
            m_label = ants.image_read(m_label_path)
            # 对moving图像对应的label图进行配准
            warped_label = ants.apply_transforms(fixed=f_img, moving=m_label, transformlist=mytx['fwdtransforms'],
                                                 interpolator="nearestNeighbor")
            warped_label.set_direction(f_img.direction)
            warped_label.set_origin(f_img.origin)
            warped_label.set_spacing(f_img.spacing)
            # 把原始的文件覆盖
            ants.image_write(warped_label, m_label_path)
    return


def registration(src_path, dst_path, ref_path):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-bins", "256", "-cost", "corratio", "-searchrx", "0", "0",
               "-searchry", "0", "0", "-searchrz", "0", "0", "-dof", "12",
               "-interp", "spline"]
    subprocess.call(command, stdout=open(os.devnull, "r"), stderr=subprocess.STDOUT)
    return

#
# def load_nii(path):
#     nii = nib.load(path)
#     return nii.get_data(), nii.affine
#
#
# def save_nii(data, path, affine):
#     nib.save(nib.Nifti1Image(data, affine), path)
#     return


def ronghe(regFilePath, m_seg_filepath, dstFilePath):
    ori_img = ants.image_read(regFilePath)
    mask_img = ants.image_read(m_seg_filepath)
    assert ori_img.shape == mask_img.shape
    print('ori_img', ori_img.shape)
    ronghe_img = np.multiply(ori_img, mask_img)
    print('ronghe_img', ronghe_img.dtype)
    #ronghe_img = np.array(ronghe_img)
    ronghe_img = ants.from_numpy(ronghe_img.numpy())
    # 这里需要把ronghe_img转换成ants的图像
    ronghe_img.set_direction(ori_img.direction)
    ronghe_img.set_origin(ori_img.origin)
    ronghe_img.set_spacing(ori_img.spacing)
    # 把原始的文件覆盖
    ants.image_write(ronghe_img, dstFilePath)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # input and anli
    input_anli_path = '/home/zhenghuadong/project/Preprocessing-Pipeline-on-Brain-MR-Images-master/data_dicom2nii/output/'
    input_seg_path = '/home/zhenghuadong/project/Preprocessing-Pipeline-on-Brain-MR-Images-master/data_from_matlab/mask_for_skull_stripping/'
    output_anli_path = '/home/zhenghuadong/project/Preprocessing-Pipeline-on-Brain-MR-Images-master/data_from_matlab/sk_reg/'


    input_anli_list = os.listdir(input_anli_path)
    for anli_dir in input_anli_list:
        anli_dir_path = os.path.join(input_anli_path, anli_dir)
        output_anli_dir_path = os.path.join(output_anli_path, anli_dir)
        if not os.path.exists(output_anli_dir_path):
            os.makedirs(output_anli_dir_path)
        # Reference every anli sample template
        refPath = os.path.join(anli_dir_path, anli_dir + '_t2.nii.gz')
        # Reference FSL sample template
        # refPath = '$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz'

        # Set ART location
        os.environ['ARTHOME'] = './utils/atra1.0_LinuxCentOS6.7/'


        niiGzPaths = Path(anli_dir_path).glob('**/*.gz')
        niiGzFiles = [niiGzPath for niiGzPath in niiGzPaths if niiGzPath.is_file()]

        # 颅脑提取的mask的标签也需要配准一下，这个mask生成是在T1图像上用matlab生成的，而我们配准需要把图像配准到T2图像上（因为标签是在T2模态下做的）
        # 具体操作是，当配准的图像是原始的T1图像时，我们就对所有的mask应用这个（T1->T2）的变换，这样就把所有的原始图像和mask都配准到T2模态下
        m_seg_dir_path = os.path.join(input_seg_path, anli_dir)
        regFiles = list()
        for niiGzFile in niiGzFiles:
            niiGzFilePath = niiGzFile.as_posix()
            # dstFile = niiGzFile.parent / (niiGzFile.stem.split('.')[0] + '_reg.nii.gz')
            dstFile = Path(output_anli_dir_path) / (os.path.basename(niiGzFile))
            regFiles.append(dstFile)
            dstFilePath = dstFile.as_posix()
            logging.info('Registration on: {}'.format(niiGzFilePath))
            try:
                # print('test')
                # orient2std(niiGzFilePath, dstFilePath)
                # registration(dstFilePath, dstFilePath, refPath)
                registration_ants_f(niiGzFilePath, m_seg_dir_path, dstFilePath, refPath)
            except RuntimeError:
                logging.warning('Falied on: {}'.format(niiGzFilePath))

        # 融合配准后的图像和mask的图像
        # 融合函数
        for regFile in regFiles:
            regFilePath = regFile.as_posix()
            # dstFile = regFile.parent / (regFile.stem.split('.')[0] + '_strip.nii.gz')
            dstFile = Path(output_anli_dir_path) / (os.path.basename(regFile))
            dstFilePath = dstFile.as_posix()
            # 对应mask文件的path
            m_seg_file = Path(m_seg_dir_path)/(regFile.stem.split('.', 2)[0] + '_seg.nii.gz')
            m_seg_filepath = m_seg_file.as_posix()
            logging.info('ronghe on : {}'.format(regFilePath))
            try:
                print('1')
                ronghe(regFilePath, m_seg_filepath, dstFilePath)
            except RuntimeError:
                logging.warning('Failed on: {}'.format(regFilePath))


