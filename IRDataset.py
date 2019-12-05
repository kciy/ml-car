import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import cv2
import numpy as np


class IRDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root_dir, transform=None):
        super(IRDataset, self).__init__(root_dir, transform)
        self.root_dir = root_dir
        self.categories = sorted(os.listdir(root_dir))
        self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
        self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
        self.files = []
        for (dirpath, dirnames, filenames) in os.walk(self.root_dir):
            for f in filenames:
                if f.endswith('.png'):
                    o = {}
                    o['img_path'] = dirpath + '/' + f
                    o['category'] = self.cat2idx[dirpath[dirpath.find('/')+1:]] # 0 and 1 as labels
                    self.files.append(o)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]['img_path']
        category = self.files[idx]['category']
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = (img.astype(np.float32) - 24000) / (21500 - 24000)
        img = np.clip(img, 0, 1)
        img_3chan = np.array([[[s,s,s] for s in r] for r in img])
        img_3chan = cv2.resize(img_3chan, (224, 224))
        img_3chan = np.transpose(img_3chan, (1,0,2))

        if self.transform is not None:
            img_3chan = self.transform(img_3chan)
        
        return {'image': img_3chan, 'category': category}
