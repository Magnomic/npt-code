
import torch.nn as nn
import torch.nn.functional as F
import models.basicblock as B
import torch

"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# IRCNN (7 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
# --------------------------------------------
"""


# --------------------------------------------
# DnCNN
# --------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, ks=3, pd=1, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias, kernel_size=ks, padding=pd) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n
    

class three_conv(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, kernel_size=3, stride=1, padding="same", dilation=1):
        super(three_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation),
                    nn.BatchNorm2d(out_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation),
                    nn.BatchNorm2d(out_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class MFF(nn.Module):
    def __init__(self, channel=16):
        super(MFF, self).__init__()

        self.conv_3x3 = three_conv(channel, channel, kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_5x5 = three_conv(channel, channel, kernel_size=5, stride=1, padding='same', dilation=1)
        self.conv_7x7 = three_conv(channel, channel, kernel_size=7, stride=1, padding='same', dilation=1)
        self.conv_fusion = nn.Sequential(
                    nn.Conv2d(channel*3, channel, kernel_size=3,stride=1, padding='same'),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, kernel_size=3,stride=1, padding='same'),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(inplace=True))
        

    def forward(self, x):
        identity = x
              
        out1 = self.conv_3x3(x)
        out2 = self.conv_5x5(x)
        out3 = self.conv_7x7(x)
        
        fusion = torch.cat([out1, out2, out3], 1)
        out = self.conv_fusion(fusion)
        
        fusionout = out - identity
        
        return fusionout




class MFFCNN(nn.Module):
    def __init__(self):
        super(MFFCNN, self).__init__()
        self.headconv = nn.Conv2d(1, 16, 3, padding="same")
        self.relu = nn.ReLU()
        self.block1 = MFF(16)
        self.block2 = MFF(16)
        self.block3 = MFF(16)
        self.tailconv = nn.Conv2d(16, 1, 3, padding="same")


    def forward(self, inputs):
        headout = self.relu(self.headconv(inputs))
        bodyout = self.block3(self.block2(self.block1(headout)))
        out = self.relu(self.tailconv(bodyout))
        return out


class ASPP_16(nn.Module):
    def __init__(self, channel=16):
        super(ASPP_16, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(channel)

        self.conv_3x3_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding="same", dilation=1)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(channel)

        self.conv_3x3_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding="same", dilation=2)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(channel)

        self.conv_3x3_3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding="same", dilation=3)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(channel)

        self.conv_1x1_3 = nn.Conv2d(channel*5, channel, kernel_size=1) 
        self.bn_conv_1x1_3 = nn.BatchNorm2d(channel)

        self.conv_1x1_4 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, channel, h, w))

        feature_map_h = feature_map.size()[2] # (== h)
        feature_map_w = feature_map.size()[3] # (== w)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))
        
        
        return out
    
class ASPPCNN(nn.Module):
    def __init__(self):
        super(ASPPCNN, self).__init__()
        self.headconv = nn.Conv2d(1, 16, 3, padding="same")
        self.relu = nn.ReLU()
        self.block1 = ASPP_16(16)
        self.block2 = ASPP_16(16)
        self.block3 = ASPP_16(16)
        self.tailconv = nn.Conv2d(16, 1, 3, padding="same")


    def forward(self, inputs):
        headout = self.relu(self.headconv(inputs))
        bodyout = self.block3(self.block2(self.block1(headout)))
        out = self.relu(self.tailconv(bodyout))
        return out


# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x-n


# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    from utils import utils_model
    import torch
    model1 = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='BR')
    print(utils_model.describe_model(model1))

    model2 = FDnCNN(in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R')
    print(utils_model.describe_model(model2))

    x = torch.randn((1, 1, 240, 240))
    x1 = model1(x)
    print(x1.shape)

    x = torch.randn((1, 2, 240, 240))
    x2 = model2(x)
    print(x2.shape)

    #  run models/network_dncnn.py
