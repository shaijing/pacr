"""
ResNet in PyTorch.



Reference
---------
Pytorch

https://blog.csdn.net/qq_45658822/article/details/129868339

https://zhuanlan.zhihu.com/p/515734064

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from typing import Type, Union, List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "ResNet_c",
    "resnet18_c",
    "resnet34_c",
    "resnet50_c",
    "resnet101_c",
    "resnet152_c",
    "SimpleCNN",
]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.fc_feat = nn.Linear(128, feat_dim)
        self.fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_feat=False):
        h = self.conv(x).view(x.size(0), -1)
        z = self.fc_feat(h)
        z = F.relu(z)
        out = self.fc_out(z)
        if return_feat:
            return out, z
        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:  # 普通Block简单完成两次卷积操作
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()  # 负责升维下采样的卷积网络
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)  # 调用shortcut对输入修改，为后面相加做变换准备

        # 完成一次卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二次卷积不加relu激活函数
        out = self.conv2(out)
        out = self.bn2(out)

        # 两路相加
        out += identity

        # 添加激活函数输出
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_c(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,  # cifar10为10
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        x = self.linear(z)
        if return_feat:
            return x, z
        return x


def resnet18_c(num_classes=10):
    return ResNet_c(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34_c(num_classes=10):
    return ResNet_c(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50_c(num_classes=10):
    return ResNet_c(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101_c(num_classes=10):
    return ResNet_c(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152_c(num_classes=10):
    return ResNet_c(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test_c():
    net = resnet18_c()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# class SpecialBlock(nn.Module):  # 特殊Block完成两次卷积操作，以及一次升维下采样
#     def __init__(self, inplanes, planes, stride):  # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
#         super(SpecialBlock, self).__init__()
#         self.change_channel = nn.Sequential(  # 负责升维下采样的卷积网络change_channel
#             nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride[0], padding=0, bias=False),
#             nn.BatchNorm2d(planes)
#         )
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride[0], padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride[1], padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#     def forward(self, x):
#         # 调用change_channel对输入修改，为后面相加做变换准备
#         identity = self.change_channel(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         # 完成残差部分的卷积
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         # 两路相加
#         out += identity
#
#         # 添加激活函数输出
#         out = self.relu(out)
#
#         # 输出卷积单元
#         return out
#
#
# class resnet18(nn.Module):
#     def __init__(self):
#         super(resnet18, self).__init__()
#
#
#         # 所有的ResNet共有的预处理==[batch, 64, 56, 56]
#         self.prepare = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1),  # 更改处
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#
#         # layer1有点特别，由于输入输出的channel均是64，故两个BasicBlock
#         self.layer1 = nn.Sequential(
#             BasicBlock(64, 64, 1),
#             BasicBlock(64, 64, 1)
#         )
#
#         # layer234类似，由于输入输出的channel不同，故一个SpecialBlock，一个BasicBlock
#         self.layer2 = nn.Sequential(
#             SpecialBlock(64, 128, [2, 1]),
#             BasicBlock(128, 128, 1)
#         )
#         self.layer3 = nn.Sequential(
#             SpecialBlock(128, 256, [2, 1]),
#             BasicBlock(256, 256, 1)
#         )
#         self.layer4 = nn.Sequential(
#             SpecialBlock(256, 512, [2, 1]),
#             BasicBlock(512, 512, 1)
#         )
#
#         # 卷积结束，通过一个自适应均值池化== [batch, 512, 1, 1]
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#         # 基于三层全连接层构成分类器网络
#         self.fc = nn.Sequential(  # 最后用于分类的全连接层，根据需要灵活变化
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, 10)  # 这个使用CIGAR10数据集，定为10分类
#         )
#
#     def forward(self, x: Tensor) -> Tensor:
#         # 预处理
#         x = self.prepare(x)
#
#         # 四个卷积单元
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)  # torch.Size([32, 512, 4, 4])
#
#         # 池化 torch.Size([32, 512, 1, 1])
#         x = self.pool(x)
#
#         # 将x展平，输入全连接层
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
#
#         return x
