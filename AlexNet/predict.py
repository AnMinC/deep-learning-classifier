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

flowerdict = dict(zip([0, 1, 2, 3, 4], ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']))

class AlexNet_Predict(torch.nn.Module):
    def __init__(self):
        super(AlexNet_Predict, self).__init__()
        self.alexNet = model.AlexNet()
        self.predict = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.alexNet(x)
        out = self.predict(out)
        return out



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net = AlexNet_Predict()
net.alexNet.load_state_dict(torch.load('alexNet.pth', map_location=torch.device('cpu')))
img = Image.open('234.jpg')
img = transform(img)
img = torch.unsqueeze(img, dim=0)
net.eval()
with torch.no_grad():
    outs = net(img)
    label = torch.argmax(outs).item()
print(f'classifier: {flowerdict[int(label)]} ')





