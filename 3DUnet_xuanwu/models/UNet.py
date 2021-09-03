import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)

        # 256*256 尺度下的映射，将256*256映射到输出尺寸上，然后计算损失（每上采样一次计算一次损失）。原始尺寸是512*512，经过数据预处理之后是256*256，体积的维度是深度
        # 所以高度和宽度方向放大一倍就行，深度方向
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),  # scale_factor=(1, 2, 2)是放大倍数
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self, x):
        # down1 = self.encoder1(x)  # 1*32*48*256*256,encoder只改变了通道数，没有改变体数据尺寸
        # print('down1.shape', down1.shape)
        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))  # 1*32*24*128*128，池化操作将体数据尺寸减半，但是这个操作参数没有看懂
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))  # 1*64*12*64*64
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))  # 1*128*6*32*32
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))  # 1*256*3*16*16
        # print('t1.shape', t1.shape, 't2.shape', t2.shape, 't3.shape', t3.shape, 'out.shape', out.shape)
        # t4 = out
        # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))
        
        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        output1 = self.map1(out)
        decode2 = self.decoder2(out)  # 1*128*3*16*16, decode就是一个卷积，加了pading,只改变通道数
        print('decode2.shape', decode2.shape)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))  # # 1*128*6*32*32，三线性插值将体数据尺寸放大2倍
        print('out.shape', out.shape)
        out = torch.add(out,t3)  # 1*128*6*32*32， 这里用的是add，就是对应元素直接相加，而不是cat,就不是原汁原味的unet
        print('out.shape', out.shape)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        print(out.shape)
        output4 = self.map4(out)

        print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4  # 这里输出的四个output是为了更好的训练，还不如直接融合一下，像FPN一样
        else:
            return output4