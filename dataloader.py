import json
import os

import torch.utils.data as data
from PIL import Image


class MyDataset(data.Dataset):
    def __init__(self, jsonfile, jpglistfile, dir_path, new_width, new_height, transform=None):
        imgs = []
        f = open(jsonfile, "r")
        d = json.load(f)
        f.close()
        with open(jpglistfile, "r") as fp:
            for line in fp:
                imgs.append((line[:-1], [d['extraversion'][line[:19]],
                                   d['neuroticism'][line[:19]],
                                   d['agreeableness'][line[:19]],
                                   d['conscientiousness'][line[:19]],
                                 #  d['interview'][line[:19]],
                                   d['openness'][line[:19]]]))
        self.imgs = imgs
        self.dir_path = dir_path
        self.height = new_height
        self.width = new_width
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        # idx = path.split('/')[1].split('.')[0]
        path = os.path.join(self.dir_path, path)
        img = Image.open(path).convert('RGB')
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
