import os
import sys
import shutil
import cv2
from PIL import Image, ImageTk
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def image_loader(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img = (img.astype(np.float32) - 24000) / (21500 - 24000)
    img = np.clip(img, 0, 1)
    img_3chan = np.array([[[s,s,s] for s in r] for r in img])
    img_3chan = cv2.resize(img_3chan, (224, 224))
    image = np.transpose(img_3chan, (1,0,2))
    return image


mean = [0.5432, 0.5432, 0.5432]
std = [0.3317, 0.3317, 0.3317]
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])

FILENAME = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/fl_rgb/fl_rgb_1566397051_9842286390/fl_rgb_1566397051_9842286390_0.ir.det.png"
MODELPATH = "/home/viki/Documents/Informatik/von ya/1sv62nza/ActiveCarModel13.pth"
model = torchvision.models.resnet18(pretrained=True).load_state_dict(torch.load(MODELPATH, map_location='cpu'))

model.eval()
print(np.argmax(model(image_loader(FILENAME)).detach().numy()))
