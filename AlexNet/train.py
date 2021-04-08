#! -*- coding:utf-8 -*-

import torch
import DataSet
import model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ]
)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    acc = 0.0
    net = model.AlexNet()
    train_loader = DataLoader(DataSet.Flower_Data('train', transform=transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(DataSet.Flower_Data('val', transform=transform), batch_size=8, shuffle=True)
    option = torch.optim.Adam(params=net.parameters(), lr=0.0002)
    lossFunction = torch.nn.CrossEntropyLoss()

    for epoch in range(20):
        running_loss = 0.0
        net.train()
        for step, img in enumerate(train_loader, 0):
            img, labels = img
            img = img.to(device)
            labels = labels.to(device)
            option.zero_grad()
            outs = net(img)
            loss = lossFunction(outs, labels)
            loss.backward()
            option.step()
            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = '*' * int(rate * 50)
            b = '.' * int((1-rate) * 50)
            print('\r train:{:^3.0f}% [{}->{}],loss: {:.3f}'.format(int(rate * 100), a, b, loss), end='')
        print('')
        steps = step + 1
        accuracy = 0.0
        net.eval()
        with torch.no_grad():
            for step, img in enumerate(val_loader, 0):
                img, labels = img
                img = img.to(device)
                labels = labels.to(device)
                outs = net(img)
                predict_y = torch.max(outs, dim=1)[1]
                accuracy += (predict_y == labels).sum().item()
        accuracy = accuracy / 346
        print(f'epoch {epoch + 1} loss is {running_loss / steps} acc is {accuracy}')
        if acc < accuracy:
            acc = accuracy
            torch.save(net.state_dict(), './alexNet.pth')







if __name__ == '__main__':
    main()