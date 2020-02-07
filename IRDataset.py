import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from random import random, randint

def rotateImage(img, angle):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


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
        img = (img.astype(np.float32) - 24000) / (21800 - 24000)
        img = np.clip(img, 0, 1)
        img_3chan = cv2.merge((img, img, img))
        # print(img_3chan.shape)
        # print(img_3chan)

        ## image augmentations
        random_rotate = random()
        random_flip_vertical = random()
        
        if random_flip_vertical > 0.5:
            img_3chan = cv2.flip(img_3chan, 1)
        if random_rotate > 0.4:
            random_angle = randint(-20, 20)
            img_3chan = rotateImage(img_3chan, random_angle)
        
        img_3chan = cv2.resize(img_3chan, (224, 224), interpolation=cv2.INTER_AREA)
        # SHOW IMAGES
        img_3chan = (img_3chan * 255).astype(np.uint8)
        img_3chan = cv2.applyColorMap(img_3chan, cv2.COLORMAP_JET)
        img_3chan = cv2.bitwise_not(img_3chan) # reverse colormap

        cv2.imshow("win", img_3chan)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

        if self.transform is not None: # ToTensor() and Normalize(mean, std)
            img_3chan = self.transform(img_3chan)
        
        return {'image': img_3chan, 'category': category, 'img_path': img_path}


class IRDatasetTest(torchvision.datasets.ImageFolder):
    def __init__(self, root_dir, transform=None):
        super(IRDatasetTest, self).__init__(root_dir, transform)
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
        img = (img.astype(np.float32) - 24000) / (21800 - 24000)
        img = np.clip(img, 0, 1)
        img_3chan = cv2.merge((img, img, img)) 
        img_3chan = cv2.resize(img_3chan, (224, 224))

        if self.transform is not None: # ToTensor() and Normalize(mean, std)
            img_3chan = self.transform(img_3chan)
        
        return {'image': img_3chan, 'category': category, 'img_path': img_path}
