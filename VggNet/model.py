#! -*- coding:utf-8 -*-

import torch
from torchsummary import summary

cfg = {
    'vgg11': [64, 'm', 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'],
    'vgg13': [64, 64, 'm', 128, 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'],
    'vgg16': [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', 512, 512, 512, 'm'],
    'vgg19': [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 256, 'm', 512, 512, 512, 512, 'm', 512, 512, 512, 512, 'm']
}

class VggNet(torch.nn.Module):
    def __init__(self, cfgs, num_class=1000):
        super(VggNet, self).__init__()
        layers = []
        self.inchannel = 3
        for cfg in cfgs:
            if cfg == 'm':
                layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            else:
                layers.extend(
                    [
                        torch.nn.Conv2d(in_channels=self.inchannel, out_channels=cfg, kernel_size=3, stride=1, padding=1),
                        torch.nn.ReLU(inplace=True),
                    ]
                )
                self.inchannel = cfg
        self.feature = torch.nn.Sequential(*layers)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=7*7*512, out_features=4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=4096, out_features=num_class),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.feature(x)
        out = out.view((-1, 7*7*512))
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    net = VggNet(cfg['vgg16'], 5)
    summary(net, (3, 224, 224))