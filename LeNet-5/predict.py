#! -*- coding:utf-8 -*-

import torch
import mode
from PIL import Image
import torchvision.transforms as transforms


transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ]
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class LeNet_5_Predict(torch.nn.Module):
    def __init__(self):
        super(LeNet_5_Predict, self).__init__()
        self.feature = mode.LeNet_5()
        self.out = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.feature(x)
        out = self.out(out)
        return out


def main():
    net = LeNet_5_Predict()
    net.feature.load_state_dict(torch.load('Lenet.pth'))
    img = Image.open('345.jpg')
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        out = net(img)
        acc, index = torch.max(out, dim=1)
    acc = acc.item()
    index = index.item()
    print("目标为{}, 概率大小为{:.5}%".format(classes[int(index)], acc*100))




if __name__ == '__main__':
    main()
