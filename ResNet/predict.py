#! -*- coding:utf-8 -*-

import torch
from PIL import Image
import model
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ]
)

class RetNet_Predict(torch.nn.Module):
    def __init__(self):
        super(RetNet_Predict, self).__init__()
        block, layers = model.cfgs['res18']
        self.res = model.RetNet(block, layers, 5)
        self.out = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.res(x)
        out = self.out(out)
        return out

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    flowerdict = dict(
        zip([0, 1, 2, 3, 4], ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
    )
    net = RetNet_Predict().to(device)
    net.res.load_state_dict(torch.load('resNet18.pth', map_location='cpu'))
    img = Image.open('123.jpg')
    img = transform(img)
    img = torch.unsqueeze(img, dim=0).to(device)
    net.eval()
    with torch.no_grad():
        outs = net(img)
    index = torch.argmax(outs).item()
    print(f'classifier is {flowerdict[int(index)]}')
