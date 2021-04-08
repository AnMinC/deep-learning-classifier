# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import mode
from torchsummary import summary
import PIL
import numpy as np
import cv2

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ]
)

trainloader = DataLoader(dataset=torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=False),
                         batch_size=32,
                         shuffle=True)

testloader = DataLoader(dataset=torchvision.datasets.CIFAR10(root='data', train=False, transform=transform, download=True),
                        batch_size=5000)

val_data_iter = iter(testloader)
val_image, val_label = val_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    # 初始化网络
    net = mode.LeNet_5()
    # 初始化优化器
    option = torch.optim.Adam(params=net.parameters(), lr=0.001)
    # 交叉墒 + softmax
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        running_loss = 0.0
        for step, data in enumerate(trainloader, 0):
            inputs, labels = data
            option.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            option.step()
            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    out = net(val_image)
                    predict_y = torch.max(out, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' % (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    torch.save(net.state_dict(), './Lenet.pth')




if __name__ == '__main__':
    main()








