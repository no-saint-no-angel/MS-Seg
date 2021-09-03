import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, encoder=True):
        super(DoubleConv3d, self).__init__()
        if encoder:
            conv1_out = out_ch//2
        else:
            conv1_out = out_ch
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, conv1_out, 3, padding=1),
            nn.BatchNorm3d(conv1_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv1_out, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
            return self.conv(input)


class UNet_standard1(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet_standard1, self).__init__()
        self.training = training
        # self.encoder1 = DoubleConv3d(in_channel, 64)
        # self.encoder2 = DoubleConv3d(64, 128)
        # self.encoder3 = DoubleConv3d(128, 256)
        self.encoder1 = nn.Conv3d(in_channel, 64, 3, padding=1)
        self.encoder2 = nn.Conv3d(64, 128, 3, padding=1)
        self.encoder3 = nn.Conv3d(128, 256, 3, padding=1)
        self.encoder4 = nn.Conv3d(256, 256, 3, padding=1)

        # self.decoder1 = DoubleConv3d(256+256, 256, encoder=False)
        self.decoder1 = nn.Conv3d(256+256, 256, 3, padding=1)
        # self.decoder2 = DoubleConv3d(128+256, 128, encoder=False)
        self.decoder2 = nn.Conv3d(128+256, 128, 3, padding=1)
        # self.decoder3 = DoubleConv3d(64+128, 64, encoder=False)
        self.decoder3 = nn.Conv3d(64+128, 64, 3, padding=1)
        # conv_out
        self.conv_out = nn.Conv3d(64, out_channel, 3, padding=1)

        # 256*256 尺度下的映射，将256*256映射到输出尺寸上，然后计算损失（每上采样一次计算一次损失）。原始尺寸是512*512，经过数据预处理之后是256*256，体积的维度是深度
        # 所以高度和宽度方向放大一倍就行，深度方向
        self.map4 = nn.Sequential(
            # nn.Conv3d(2, out_channel, 1, 1),
            # nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),  # scale_factor=(1, 2, 2)是放大倍数
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self, x):
        # down1 = self.encoder1(x)  # 1*32*48*256*256,encoder只改变了通道数，没有改变体数据尺寸
        # print('down1.shape', down1.shape)
        # encode
        out = self.encoder1(x)  # 1*64*48*259*256
        # print(out.shape)
        t1 = out

        out = F.relu(F.max_pool3d(out, 2, 2))  # 1*64*24*128*128，池化操作将体数据尺寸减半，但是这个操作参数没有看懂
        out = self.encoder2(out)
        t2 = out

        out = F.relu(F.max_pool3d(out, 2, 2))  # 1*128*12*64*64
        out = self.encoder3(out)
        t3 = out

        out = F.relu(F.max_pool3d(out, 2, 2))  # 1*256*6*32*32
        # print(t1.shape, t2.shape, t3.shape)
        out = self.encoder4(out)  # 1*512*6*32*32
        # print(out.shape)
        output1 = self.map1(out)
        # print(output1.shape)

        # decode
        out = F.relu(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # # 1*512*12*64*64，三线性插值将体数据尺寸放大2倍
        out = self.decoder1(torch.cat([out, t3], dim=1))  # 1*(256+512)*12*64*64->256*12*64*64
        output2 = self.map2(out)

        out = F.relu(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # 1*(256)*24*128*128
        out = self.decoder2(torch.cat([out, t2], dim=1))  # 1*(128+256)*24*128*128->1*128*24*128*128
        output3 = self.map3(out)

        out = F.relu(F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear'))  # 1*128*48*256*256
        out = self.decoder3(torch.cat([out, t1], dim=1))  # 1*(64+128)*48*256*256->1*64*48*256*256
        out = self.conv_out(out)
        output4 = self.map4(out)

        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            # return self.map4(out)  #
            return output1, output2, output3, output4  # 这里输出的四个output是为了更好的训练，还不如直接融合一下，像FPN一样
        else:
            return output4