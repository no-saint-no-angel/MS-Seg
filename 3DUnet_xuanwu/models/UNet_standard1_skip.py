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
            nn.InstanceNorm3d(conv1_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv1_out, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
            return self.conv(input)


class DoubleConv3d_output(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3d_output, self).__init__()
        conv1_out = 64
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, conv1_out, 3, padding=1),
            nn.InstanceNorm3d(conv1_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv1_out, out_ch, 3, padding=1),
            # nn.BatchNorm3d(out_ch),
            # nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
            return self.conv(input)


class SingleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
            return self.conv(input)

class SkipConv(nn.Module):
    def __init__(self, in_ch):
        super(SkipConv, self).__init__()
        out_ch = round(in_ch/4)
        self.project = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.project(input)


class UNet_standard1_skip(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet_standard1_skip, self).__init__()
        self.training = training
        # self.encoder1 = DoubleConv3d(in_channel, 64)
        # self.encoder2 = DoubleConv3d(64, 128)
        # self.encoder3 = DoubleConv3d(128, 256)
        self.encoder1 = SingleConv3d(in_channel, 64)
        self.encoder2 = SingleConv3d(64, 128)
        self.encoder3 = SingleConv3d(128, 256)
        self.encoder4 = SingleConv3d(256, 256)

        # self.decoder1 = DoubleConv3d(256+256, 256, encoder=False)
        self.decoder1 = SingleConv3d(round(256/4)+256, 256)
        # self.decoder2 = DoubleConv3d(128+256, 128, encoder=False)
        self.decoder2 = SingleConv3d(round(128/4)+256, 128)
        # self.decoder3 = DoubleConv3d(64+128, 64, encoder=False)
        self.decoder3 = SingleConv3d(round(64/4)+128, 64)

        # 跳跃连接采用1*1*1卷积而不是直接copy然后concatenate,
        self.skip_low = SkipConv(64)
        self.skip_mid = SkipConv(128)
        self.skip_high = SkipConv(256)

        # 对融合特征图进行双卷积操作
        self.doubleconv_out = DoubleConv3d_output(64+128+256, out_channel)
        # 256*256 尺度下的映射，将256*256映射到输出尺寸上，然后计算损失（每上采样一次计算一次损失）。原始尺寸是512*512，经过数据预处理之后是256*256，体积的维度是深度
        # 所以高度和宽度方向放大一倍就行，深度方向
        self.map4_loss = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            # nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear'),  # scale_factor=(1, 2, 2)是放大倍数
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3_loss = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2_loss = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1_loss = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 融合特征
        self.map4 = nn.Sequential(
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
        )

    def forward(self, x):
        # down1 = self.encoder1(x)  # 1*32*48*256*256,encoder只改变了通道数，没有改变体数据尺寸
        # print('down1.shape', down1.shape)
        # encode
        out = self.encoder1(x)  # 1*64*48*259*256
        # print(out.shape)
        t1 = self.skip_low(out)

        out = F.max_pool3d(out, 2, 2)  # 1*64*24*128*128，池化操作将体数据尺寸减半，但是这个操作参数没有看懂
        out = self.encoder2(out)
        t2 = self.skip_mid(out)

        out = F.max_pool3d(out, 2, 2)  # 1*128*12*64*64
        out = self.encoder3(out)
        t3 = self.skip_high(out)

        out = F.max_pool3d(out, 2, 2)  # 1*256*6*32*32
        # print(t1.shape, t2.shape, t3.shape)
        out = self.encoder4(out)  # 1*512*6*32*32
        # print(out.shape)
        output1_loss = self.map1_loss(out)
        # print(output1.shape)

        # decode
        out = F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear')  # # 1*512*12*64*64，三线性插值将体数据尺寸放大2倍
        out = self.decoder1(torch.cat([out, t3], dim=1))  # 1*(256+512)*12*64*64->256*12*64*64
        output2_loss = self.map2_loss(out)
        output2 = self.map2(out)

        out = F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear')  # 1*(256)*24*128*128
        out = self.decoder2(torch.cat([out, t2], dim=1))  # 1*(128+256)*24*128*128->1*128*24*128*128
        output3_loss = self.map3_loss(out)
        output3 = self.map3(out)

        out = F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear')  # 1*128*48*256*256
        out = self.decoder3(torch.cat([out, t1], dim=1))  # 1*(64+128)*48*256*256->1*64*48*256*256
        output4_loss = self.map4_loss(out)

        # decode支路上不同分辨率特征图的融合
        fusion_feature = self.doubleconv_out(torch.cat([out, output3, output2], dim=1))
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            # return self.map4(out)  #
            # return fusion_feature
            return output1_loss, output2_loss, output3_loss, output4_loss, fusion_feature  # 这里输出的四个output是为了更好的训练，还不如直接融合一下，像FPN一样
        else:
            return fusion_feature