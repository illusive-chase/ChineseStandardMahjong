# -*- coding: utf-8 -*-
__all__ = ('resnet16', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'Slider')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_bn=True):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        ) if use_bn else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=True),
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            ) if use_bn else nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, use_bn=True):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        ) if use_bn else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=True),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            ) if use_bn else nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=True)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_input_channels, num_classes, use_bn, dropout):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) if use_bn else nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, use_bn)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, use_bn)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, use_bn)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, use_bn)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, use_bn):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_bn))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc(output)
        return output

class SimpleResNet(nn.Module):

    def __init__(self, block, num_block, num_input_channels, num_classes, use_bn, dropout):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) if use_bn else nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block, 1, use_bn)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 36 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, use_bn):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_bn))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc(output)
        return output


def resnet16(use_bn, dropout=0.5, shape=(145, 235)):
    return SimpleResNet(BasicBlock, 7, num_input_channels=shape[0], num_classes=shape[1], use_bn=use_bn, dropout=dropout)


def resnet18(use_bn, dropout=0.5, shape=(145, 235)):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_input_channels=shape[0], num_classes=shape[1], use_bn=use_bn, dropout=dropout)

def resnet34(use_bn, dropout=0.5, shape=(145, 235)):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_input_channels=shape[0], num_classes=shape[1], use_bn=use_bn, dropout=dropout)

def resnet50(use_bn, dropout=0.5, shape=(145, 235)):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_input_channels=shape[0], num_classes=shape[1], use_bn=use_bn, dropout=dropout)

def resnet101(use_bn, dropout=0.5, shape=(145, 235)):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_input_channels=shape[0], num_classes=shape[1], use_bn=use_bn, dropout=dropout)

def resnet152(use_bn, dropout=0.5, shape=(145, 235)):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_input_channels=shape[0], num_classes=shape[1], use_bn=use_bn, dropout=dropout)


'''
single tile: 34
double tile: 34(0), 24(1), 21(2), 18(3)
triple tile: 34(00), 21(11), 15(22), 9(33), 24(01), 24(10), 21(02), 21(20)
triple tile as pack: ...
quadra tile: 34(000)
quadra tile as pack: 34(000)

V*(s) = max[a]: sum[s_]: p(s_|s, a) V*(s_)
'''

def MLP(*nums):
    layers = []
    assert len(nums) > 1
    for in_channels, out_channels in zip(nums[:-1], nums[1:]):
        layers.append(nn.Conv1d(in_channels, out_channels, 1, padding=0))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Slider(nn.Module):

    class SlideLayer2(nn.Module):
        def __init__(self, stride):
            super().__init__()
            self.layer = MLP(64, 32, 32)
            self.stride = stride

        def forward(self, x):
            # x: b x 32 x n
            return self.layer(torch.cat((x[:, :, :-self.stride], x[:, :, self.stride:]), dim=1))

    class SlideLayer3(nn.Module):
        def __init__(self, stride1, stride2):
            super().__init__()
            self.layer = MLP(96, 64, 32)
            self.strides = (stride1, 9 - stride2, stride1 + stride2)

        def forward(self, x):
            return self.layer(torch.cat((x[:, :, :-self.strides[2]], x[:, :, self.strides[0]:self.strides[1]], x[:, :, self.strides[2]:]), dim=1))



    def __init__(self):
        super().__init__()
        # tile cube: 145 x 4 x 9 -> 32 x 34
        # slide 1, 2, 3: 32 x (8, 7, 6) = 32 x 21
        # silde 11, 22, 33, 01, 10, 02, 20: 32 x (7, 5, 3, 8, 8, 7, 7) = 32 x 45

        self.encoder = MLP(145, 256, 128, 64, 32)
        slides = ['1', '2', '3', '11', '22', '33', '01', '10', '02', '20']
        slide_layers = []
        for slide in slides:
            slide = list(map(int, slide))
            if len(slide) == 1:
                slide_layers.append(self.SlideLayer2(*slide))
            elif len(slide) == 2:
                slide_layers.append(self.SlideLayer3(*slide))
            else:
                raise ValueError

        self.slide_layers = nn.ModuleList(slide_layers)
        self.projection = MLP(32, 4)
        self.linear = nn.Sequential(
            nn.Linear(928, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 235)
        )

    def forward(self, x):
        feat = self.encoder(x.view(-1, 145, 36)[:, :, :34])
        # feat: b x 32 x 34
        m, p, s, z = feat[:, :, 0:9], feat[:, :, 9:18], feat[:, :, 18:27], feat[:, :, 27:34]
        
        results = [z]
        for t in (m, p, s):
            results.append(t)
            for layer in self.slide_layers:
                results.append(layer(t))
            # result: b x 32 x (9, 21, 45) = b x 32 x 75
        # results: b x 32 x (7, 75, 75, 75) = b x 32 x 232
        x = self.projection(torch.cat(results, dim=2))
        # x: b x 4 x 232 = b x 928 x 1
        return self.linear(x.view(x.size(0), -1))





