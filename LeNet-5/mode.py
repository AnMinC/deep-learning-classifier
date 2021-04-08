#! -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F



class LeNet_5(torch.nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.conv1_relu = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, bias=False)
        self.conv2_relu = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(in_features=32 * 5 * 5, out_features=120)
        self.fc1_relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc2_relu = torch.nn.ReLU(inplace=True)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1_relu(self.conv1(x))
        out = self.pool1(out)
        out = self.conv2_relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1, 32 * 5 * 5)
        out = self.fc1_relu(self.fc1(out))
        out = self.fc2_relu(self.fc2(out))
        out = self.fc3(out)
        return out