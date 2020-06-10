import torch
import random
import glob
import os
import sys
from PIL import Image
import pdb
from torch.utils.data import DataLoader
from torchvision import transforms


class IMDBWIKI(torch.utils.data.Dataset):

    def __init__(self, root, train=True, transform=None, dp=False, seed=0):
        random.seed(seed)

        self.root = root
        self.train = train
        self.transform = transform
        self.dp = dp

        self.images = []
        for i, path in enumerate(glob.glob(os.path.abspath('{}/*.png'.format(self.root)))):
            self.images.append(path)

        random.shuffle(self.images)
        if dp:
            self.images = self.images[-10:]
        elif train:
            self.images = self.images[:-200]
        else:
            self.images = self.images[-200:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        label = os.path.basename(path).split('_')[1].split('.')[0]

        image = Image.open(path)
        if self.transform:
            image = self.transform(image)

        return image, label


dats=IMDBWIKI('astrdataset')
tsloader = torch.utils.data.DataLoader(dats)


