#! -*- coding:utf-8 -*-

from torch.utils.data import Dataset
import os
from PIL import Image


class Flower_Data(Dataset):
    flowerdict = dict(
        zip(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'], [0, 1, 2, 3, 4])
    )
    def __init__(self, model, transform=None):
        super(Flower_Data, self).__init__()
        self.model = model
        self.data = []
        self.transform = transform
        self.bastdir = 'data_set/flower_data'
        if self.model == 'train':
            self.datadir = os.path.join(self.bastdir, 'train')
        else:
            self.datadir = os.path.join(self.bastdir, 'val')
        for classesdir in os.listdir(self.datadir):
            imgdir = os.path.join(self.datadir, classesdir)
            for imgname in os.listdir(imgdir):
                imgpath = os.path.join(imgdir, imgname)
                self.data.append([imgpath, self.flowerdict[classesdir]])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        imgpath, label = self.data[item]
        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

