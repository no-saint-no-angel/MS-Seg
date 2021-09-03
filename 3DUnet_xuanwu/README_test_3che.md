## 介绍
这个文件的功能是，介绍使用3DUnet网络对T1、T2、FLAIR三个模态的颅脑MRI进行病灶推断以及结果展示操作。
### 环境  
```angular2
pytorch >= 1.1.0
torchvision
SimpleITK
Tensorboard
Scipy
```
### 所使用到的文件结构如下
```angular2
├── dataset        # 测试集数据预处理文件
│   └── dataset_lits_test_gai_2_3che.py 
├── experiments    # 实验
|   └── UNet_standard_thin_LR0002_pre_3chaweight_3che  
|       ├── result_T2_1   # 推断结果
|       └── test_log.csv  # 推断准确率
|── fixed_data     # 预处理之后的T2数据
|   ├── ct
|   └── label
|── raw_dataset    # 原始输入数据，每个案例文件夹下包含多个模态的数据和label
|   └── test_3cha
│       ├── 01016SACH
│       ├── 01038PAGU
|       |—— ...
├── conifg.py      # 参数设置文件
└── test_3cha.py   # 推断执行文件     
```
###  测试 3DUnet
1、参数设置  
- 进入config.py文件，将'--test_data_path'参数设置成原始输入数据地址'./raw_dataset/test_3cha'；
- '--save'参数，训练参数保存所在的地址，设置成'UNet_standard_thin_LR0002_pre_3chaweight_3che'；
  
2、结果推断
- 执行 `test_3cha.py`，会在'./experiments/UNet_standard_thin_LR0002_pre_3chaweight_3che'文件夹下生成
推断结果（/result_T2_1）和推断结果的准确率文件（test_log.csv），其中'/result_T2_1'文件名在test_3cha.py中的result_save_path设置。

3、结果展示

分割结果选择在T2序列下使用ITK-SNAP软件展示：
- 标签结果展示：首先打开'fixed_data/ct'文件夹下的T2图像，再打开'fixed_data/label'文件夹下和T2图像对应的标签图像；
- 模型推断结果展示：选择和标签结果展示中相同的T2图像，再打开'./experiments/UNet_standard_thin_LR0002_pre_3chaweight_3che/result_T2_1'文件夹下和T2图像对应的标签图像。
