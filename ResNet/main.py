#! -*- coding:utf-8 -*-

import torch
import model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import DataSet


transformdict = {
    'train': transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ]
    )

}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    block, layers = model.cfgs['res18']
    net = model.RetNet(block, layers, 5).to(device)
    train_loader = DataLoader(DataSet.Flower_Data('train', transform=transformdict['train']), batch_size=16,shuffle=True)
    val_loader = DataLoader(DataSet.Flower_Data('val', transform=transformdict['val']), batch_size=8, shuffle=True)
    option = torch.optim.Adam(params=net.parameters(), lr=0.0002)
    loss_func = torch.nn.CrossEntropyLoss()
    acc = 0.0
    for epoch in range(20):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, 0):
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            option.zero_grad()
            outs = net(img)
            loss = loss_func(outs, labels)
            loss.backward()
            option.step()
            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = '*' * int(rate * 50)
            b = '.' * int((1 - rate) * 50)
            print('\r train:{:^3.0f}% [{}->{}],loss: {:.3f}'.format(int(rate * 100), a, b, loss), end='')
        print('')
        net.eval()
        steps = step + 1
        accuracy = 0.0
        for step, data in enumerate(val_loader, 0):
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            outs = net(img)
            predict_y = torch.max(outs, dim=1)[1]
            accuracy += (predict_y == labels).sum().item()

        accuracy = accuracy / 346
        print(f'epoch {epoch + 1} loss is {running_loss / steps} acc is {accuracy}')
        if acc < accuracy:
            acc = accuracy
            torch.save(net.state_dict(), './resNet18.pth')


if __name__ == '__main__':
    main()
