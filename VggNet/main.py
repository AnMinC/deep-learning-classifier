#! -*- coding:utf-8 -*-

import torch
from torch.utils.data import DataLoader
import model
import torchvision.transforms as transforms
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
    net = model.VggNet(model.cfg['vgg16'], 5).to(device)
    option = torch.optim.Adam(params=net.parameters(), lr=0.0002)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(DataSet.Flower_Data('train', transform=transformdict['train']), batch_size=16, shuffle=True)
    val_loader = DataLoader(DataSet.Flower_Data('val', transform=transformdict['val']), batch_size=8, shuffle=True)
    acc = 0.0
    for epoch in range(20):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, 0):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            option.zero_grad()
            out = net(img)
            loss = loss_func(out, label)
            loss.backward()
            option.step()
            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = '*' * int(rate * 50)
            b = "." * int((1-rate) * 50)
            print('\r train:{:^3.0f}% [{}->{}],loss: {:.3f}'.format(int(rate * 100), a, b, loss), end='')
        print('')
        net.eval()
        steps = step + 1
        accuracy = 0.0
        with torch.no_grad():
            for step, data in enumerate(val_loader, 0):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                outs = net(img)
                predict_y = torch.max(outs, dim=1)[1]
                accuracy += (predict_y == label).sum().item()
        accuracy = accuracy / 346
        print(f'epoch {epoch + 1} loss is {running_loss / steps} acc is {accuracy}')
        if acc < accuracy:
            acc = accuracy
            torch.save(net.state_dict(), './vggNet.pth')



if __name__ == '__main__':
    main()


