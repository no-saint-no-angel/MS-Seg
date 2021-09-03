from dataset.dataset_lits_val_3che import Val_Dataset
from dataset.dataset_lits_train_3che import Train_Dataset
from data_parallel import BalancedDataParallel
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from models import UNet, ResUNet, KiUNet_min, SegNet, UNet_standard1, UNet_standard1_skip_double,vnet

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict


def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            # data, target = data.to(device), target.to(device)
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss=loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels==3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target,n_labels)
        # data, target = data.to(device), target.to(device)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # print('data.shape', data.shape)
        output = model(data)
        # print('output.shape', output.shape)
        loss = loss_func(output, target)
        # loss1 = loss_func(output[1], target)
        # loss2 = loss_func(output[2], target)
        # loss3 = loss_func(output[3], target)
        # loss4 = loss_func(output[4], target)
        # loss = loss4 + alpha * (loss0 + loss1 + loss2 + loss3)
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(),data.size(0))
        train_dice.update(output, target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels==3: val_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return val_log

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    # device = torch.device('cpu' if args.cpu else 'cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args),batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args),batch_size=1,num_workers=args.n_threads, shuffle=False)

    # model info
    # model = ResUNet(in_channel=1, out_channel=args.n_labels,training=True).to(device)
    model = UNet_standard1_skip_double(in_channel=3, out_channel=args.n_labels, training=True).cuda()
    #model = vnet.VNet(elu=False, nll=False).cuda()
    # model.apply(weights_init.init_model)
    # model.apply(weights_init.init_weights(model, init_type='kaiming'))
    # # pre_weight
    model_pre_weight = "./pre_weights/best_model_UNet_standard1_skip_double_beta_LR0002_48_pre_hengduan.pth"
    pretrained_dict = torch.load(model_pre_weight)
    pretrained_dict = pretrained_dict['net']
    #model = model.load_state_dict(pretrained_dict)
    model_dict = model.state_dict()  # pytorch自动初始�?
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # �������е�model_dict
    model_dict.update(pretrained_dict)
    # ��������������Ҫ��state_dict
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    common.print_network(model)
    #torch.cuda.set_device(1)
    #model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    #model = torch.nn.DataParallel(model, device_ids=[1, 2])  # multi-GPU
    loss = loss.TverskyLoss()
    # loss = loss.DiceLoss()

    log = logger.Train_Logger(save_path,"train_log")

    best = [0,0] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数�?
    alpha = 0.4 # 深监督衰减系数初始�?
    gpu0_bsz = 6
    acc_grad = 2
    # GPUs --------------------------------------------------
    model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model_UNet_standard1_skip_double_beta_LR0003_48.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        # 深监督系数衰�?
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()    