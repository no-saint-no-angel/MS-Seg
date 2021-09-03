from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config_test
from utils import logger,common
from dataset.dataset_lits_test_gai_2_3che_xuanwu import Test_Datasets,to_one_hot_3d
import SimpleITK as sitk
import os
import numpy as np
from models import ResUNet, UNet_standard1, UNet_standard1_skip_double
from utils.metrics import DiceAverage
from collections import OrderedDict
from plot_3d import plot_3d


def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)
    print('target.shape', target.shape)
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            print('data.shape', data.shape)
            data = data.to(device)
            output = model(data)
            #output = torch.nn.Softmax(dim=1)(output)
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred, dim=1)

    pred_img = common.to_one_hot_3d(pred,args.n_labels)
    test_dice.update(pred_img, target)
    
    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels==3: test_dice.update({'Dice_tumor': test_dice.avg[2]})
    
    pred = np.asarray(pred.numpy(),dtype='uint8')
    if args.postprocess:
        pass # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred,axis=0))

    return test_dice, pred


if __name__ == '__main__':
    args = config_test.args
    #save_path = os.path.join('./experiments', args.save)
    save_path = '/data/xuwenyi/experiments/seg_unet_2108271038_hd_0.45/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = UNet_standard1_skip_double(in_channel=3, out_channel=args.n_labels,training=False).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    ckpt = torch.load('{}/best_model_UNet_standard1_skip_double_beta_LR0003_48_08251529.pth'.format(save_path), map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    result_save_path = '{}/result_T2_1'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    datasets = Test_Datasets(args.test_data_path,args=args)
    for img_dataset,file_idx in datasets:
        try:
            test_dice,pred_img = predict_one_img(model, img_dataset, args)
            # plot_3d(pred_img, 100)
            test_log.update(file_idx, test_dice)
            sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx+'.nii.gz'))
        except Exception as e:
            pass
        continue