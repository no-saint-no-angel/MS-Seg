import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, encoder=True, downsample=True):
        super(DoubleConv3d, self).__init__()
        conv_stride = 1
        if encoder:
            conv1_out = out_ch//2
        else:
            conv1_out = out_ch
        if downsample:
            conv1_stride = 2
        else:
            conv1_stride = conv_stride
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, conv1_out, 3, stride=conv1_stride, padding=1),
            nn.InstanceNorm3d(conv1_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv1_out, out_ch, 3, stride=conv_stride, padding=1),
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
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
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


class UNet_standard1_skip_double_beta(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet_standard1_skip_double_beta, self).__init__()
        self.training = training
        # self.encoder1 = DoubleConv3d(in_channel, 64)
        # self.encoder2 = DoubleConv3d(64, 128)
        # self.encoder3 = DoubleConv3d(128, 256)
        self.doubleconv_in = DoubleConv3d(in_channel, 32, downsample=False)
        self.encoder1 = DoubleConv3d(32, 64, downsample=True)
        self.encoder2 = DoubleConv3d(64, 128, downsample=True)
        self.encoder3 = DoubleConv3d(128, 256, downsample=True)
        self.encoder4 = DoubleConv3d(256, 320, downsample=True)
        # self.encoder5 = DoubleConv3d(320, 320, downsample=True)

        # self.decoder1 = DoubleConv3d(256+256, 256, encoder=False)
        # upsample
        self.up1 = nn.ConvTranspose3d(320, 320, 2, stride=2, padding=0)
        # self.decoder1 = SingleConv3d(320+64, 256)
        self.decoder1 = DoubleConv3d(320+64, 256, encoder=False, downsample=False)

        self.up2 = nn.ConvTranspose3d(256, 256, 2, stride=2, padding=0)
        # self.decoder2 = SingleConv3d(256+32, 128)
        self.decoder2 = DoubleConv3d(256+32, 128, encoder=False, downsample=False)

        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2, padding=0)
        self.decoder3 = SingleConv3d(128+16, 64)
        # self.decoder3 = DoubleConv3d(128+16, 64, encoder=False, downsample=False)

        self.up4 = nn.ConvTranspose3d(64, 64, 2, stride=2, padding=0)
        self.decoder4 = SingleConv3d(64+8, 32)
        # self.decoder4 = DoubleConv3d(64+8, 32, encoder=False, downsample=False)

        # 跳跃连接采用1*1*1卷积而不是直接copy然后concatenate,
        self.skip_low = SkipConv(32)
        self.skip_mid1 = SkipConv(64)
        self.skip_mid2 = SkipConv(128)
        self.skip_high = SkipConv(256)

        # 对融合特征图进行双卷积操作
        self.doubleconv_out = DoubleConv3d_output(32+64+128+256, out_channel)

        # 融合特征
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
        out = self.doubleconv_in(x)
        # print('doubleconv_in shape:', out.shape)
        skip_low = self.skip_low(out)
        # print('skip_low shape:', skip_low.shape)
        # encode
        out = self.encoder1(out)
        skip_mid1 = self.skip_mid1(out)

        out = self.encoder2(out)
        skip_mid2 = self.skip_mid2(out)

        out = self.encoder3(out)
        skip_high = self.skip_high(out)

        out = self.encoder4(out)

        # decode
        # print('up1 and skip_high shape:', out.shape, skip_high.shape)
        out = self.up1(out)
        # print('up1 and skip_high shape:', out.shape, skip_high.shape)
        out = self.decoder1(torch.cat([out, skip_high], dim=1))
        output1 = self.map1(out)

        out = self.up2(out)
        out = self.decoder2(torch.cat([out, skip_mid2], dim=1))
        output2 = self.map2(out)

        out = self.up3(out)
        out = self.decoder3(torch.cat([out, skip_mid1], dim=1))
        output3 = self.map3(out)

        # print('up1 and skip_low shape:', out.shape, skip_low.shape)
        out = self.up4(out)
        # print('up1 and skip_low shape:', out.shape, skip_low.shape)
        out = self.decoder4(torch.cat([out, skip_low], dim=1))

        # decode支路上不同分辨率特征图的融合
        fusion_feature = self.doubleconv_out(torch.cat([out, output3, output2, output1], dim=1))

        return fusion_feature