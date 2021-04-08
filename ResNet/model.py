#! -*- coding:utf-8 -*-

import torch
from torchsummary import summary

class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=out_channel)
        self.conv1_relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=out_channel)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identry = x
        if self.downsample is not None:
            identry = self.downsample(x)
        out = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_bn(self.conv2(out))
        out += identry
        out = self.relu(out)
        return out

class Bottleneck(torch.nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(out_channel)
        self.conv1_relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2_bn = torch.nn.BatchNorm2d(out_channel)
        self.conv2_relu = torch.nn.ReLU(inplace=True)

        self.conv3 = torch.nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=1, bias=False)
        self.conv3_bn = torch.nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identry = x
        if self.downsample is not None:
            identry = self.downsample(x)
        out = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_relu(self.conv2_bn(self.conv2(out)))
        out = self.conv3_bn(self.conv3(out))
        out += identry
        out = self.relu(out)
        return out


class RetNet(torch.nn.Module):
    def __init__(self, block, layer, num_classes):
        super(RetNet, self).__init__()
        self.inchannel = 64
        self.conv1 = torch.nn.Conv2d(3, self.inchannel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(self.inchannel)
        self.conv1_relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, block, layer[0])
        self.layer2 = self.make_layer(128, block, layer[1], 2)
        self.layer3 = self.make_layer(256, block, layer[2], 2)
        self.layer4 = self.make_layer(512, block, layer[3], 2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


    def make_layer(self, inchannel, block, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inchannel != inchannel * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inchannel, inchannel * block.expansion, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(inchannel * block.expansion)
            )
        layer = [block(self.inchannel, inchannel, stride, downsample)]
        self.inchannel = inchannel * block.expansion
        for _ in range(1, blocks):
            layer.append(block(self.inchannel, inchannel))
        return torch.nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

cfgs = {
    'res18': [BasicBlock, [2, 2, 2, 2]],
    'res34': [BasicBlock, [3, 4, 6, 3]],
    'res50': [Bottleneck, [3, 4, 6, 3]],
    'res101': [Bottleneck, [3, 4, 23, 3]],
    'res152': [Bottleneck, [3, 8, 36, 3]],
}



if __name__ == '__main__':
    net = RetNet(BasicBlock, [2, 2, 2, 2], 5)
    summary(net, (3, 224, 224))




