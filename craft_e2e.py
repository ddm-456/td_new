"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init
from torchutil import *

from basenet.vgg16_bn import vgg16_bn

class conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        # self.net = vgg16_bn(pretrained, freeze)
        # self.net.load_state_dict(copyStateDict(torch.load('vgg16_bn-6c64b313.pth')))
        # self.basenet = self.net
        self.basenet = vgg16_bn(pretrained, freeze)
        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )



        # recognition model
        self.rec_conv1 = conv3x3(128, 256)
        self.rec_conv2 = conv3x3(256, 256)
        self.rec_pool1 = nn.Maxpool2d(2, 2)
        self.rec_conv3 = conv3x3(256, 256)
        self.rec_conv4 = conv3x3(256, 256)
        self.rec_pool2 = nn.Maxpool2d(2, 2)
        self.rec_conv5 = conv3x3(256+1024, 256)


        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)
        h_fc7, h_relu5_3, h_relu4_3, h_relu3_3, h_relu2_2 = out
        '''
        # 48 48
         48 48
         96 96
         192 192
         394 394

        '''


        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)


        # recognition feature
        y_rec = self.rec_conv1(h_relu2_2)
        y_rec = self.rec_conv2(y_rec)
        y_rec = self.rec_pool1(y_rec)
        y_rec = self.rec_conv3(y_rec)
        y_rec = self.rec_conv4(y_rec)
        y_rec = self.rec_pool2(y_rec)
        y_rec = torch.cat([y_rec, h_fc7])
        y_rec = self.rec_conv5(y_rec)

        return y.permute(0, 2, 3, 1), feature, y_rec


if __name__ == '__main__':
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)
