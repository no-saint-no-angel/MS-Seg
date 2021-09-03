
## 介绍
这个文件极简记录了，数据预处理和模型训练及推断的文件使用方式。所使用的环境均为zhenghuadong目录下的环境
### 数据预处理
预处理分为两部分，预处理1是使用专用的医学图像处理工具，对图像进行配准，颅骨剔除，偏差校正等操作，预处理2是根据训练的需要进行重采样，裁剪等操作
#### 前预处理
dicom-->.nii  
项目文件夹：./Preprocessing-Pipeline-on-Brain-MR-Images-master  
环境：brainprep  
命令：python dicom2nii_multi_s_only3
#### 预处理1
执行顺序依次为：  
项目文件夹：./Preprocessing-Pipeline-on-Brain-MR-Images-master  
1、配准：  
环境：brainprep  
命令：python3 main_only_reg_ants  
2、颅骨剔除：  
环境：matlab  
3、偏差校正和去噪  
环境：brainprep  
命令：python3 main_N4_eh
#### 预处理2
裁剪等操作  
项目文件夹：./3DUNet_xuanwu  
环境：pytorch  
命令：python preprocess_brain_MRI_multi_brats19_final
### 训练
项目文件夹：./3DUNet_xuanwu  
环境：pytorch  
命令：python train_skip_double
### 推断
项目文件夹：./3DUNet_xuanwu  
环境：pytorch  
命令：python test_3che