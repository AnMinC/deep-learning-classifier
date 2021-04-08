#! -*- coding:utf-8 -*-

import torch
import torchvision.transforms as transforms
import model
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ]
)

class VggNet_Predict(torch.nn.Module):
    def __init__(self):
        super(VggNet_Predict, self).__init__()
        self.vgg = model.VggNet(model.cfg['vgg16'], 5)
        self.out = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.vgg(x)
        out = self.out(out)
        return out

if __name__ == '__main__':
    flowerdict = dict(
        zip([0, 1, 2, 3, 4], ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
    )
    net = VggNet_Predict()
    net.vgg.load_state_dict(torch.load('vggNet.pth', map_location='cpu'))
    net.eval()
    img = Image.open('234.jpg')
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        out = net(img)

    index = torch.argmax(out).item()
    print(f'classifier is {flowerdict[int(index)]}')
















