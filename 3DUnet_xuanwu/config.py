import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=list,default=[0, 1, 2, 3], help='use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=2,help='number of classes') # 分割肝脏则置�?（二类分割），分割肝脏和肿瘤则置�?（三类分割）
parser.add_argument('--upper', type=int, default=200, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
parser.add_argument('--norm_factor', type=float, default=1.0, help='')
parser.add_argument('--expand_slice', type=int, default=8, help='')
parser.add_argument('--min_slices', type=int, default=16, help='')
parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
parser.add_argument('--valid_rate', type=float, default=0.2, help='')

# data in/out and dataset
parser.add_argument('--dataset_path',default = './fixed_data_3cha_onlyms/',help='fixed trainset root path')
parser.add_argument('--test_data_path',default = '/home/zhenghuadong/project/Preprocessing-Pipeline-on-Brain-MR-Images-master/data/anli_for_windows_test/anli_prep_reg_sk_N4_eh_test_for_windows/',help='Testset path')
# parser.add_argument('--save',default='UNet_standard1_skip_double_LR0003_16_onlymask_256_811',help='save path of trained model')
parser.add_argument('--save',default='vnet/test2',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=3,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=200, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=20, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=16)
parser.add_argument('--val_crop_max_size', type=int, default=16)

# test
parser.add_argument('--test_cut_size', type=int, default=40, help='size of sliding window')
parser.add_argument('--test_cut_stride', type=int, default=20, help='stride of sliding window')
parser.add_argument('--postprocess', type=bool, default=False, help='post process')


args = parser.parse_args()


# watch -n 5 nvidia-smi