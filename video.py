import os
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision
import cv2
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from IRDataset import IRDatasetTest


def rotateImage(img, angle):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def get_rgbs():
    # get sequence of rgb full images
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/fl_rgb"
    folders = [x[0] for x in os.walk(src)]
    root = src.replace("fl_rgb", "")
    rgb_names = [root + "fl_rgb/" + x.split("/")[-1] + '.png' for x in folders]
    rgb_names = rgb_names[1:]

    return rgb_names

def get_coords(path):
    # return coordinates of all cars in an image given its path
    folder = path[:-4] # remove ".png"
    txt_files = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            txt_files.append(os.path.join(folder, file))

    coords = []
    for txt in txt_files:
        txt = open(txt, "r")
        line = list(txt)[0]
        x1 = int(line.split(" ")[2])
        y1 = int(line.split(" ")[3])
        x2 = int(line.split(" ")[4])
        y2 = int(line.split(" ")[5])
        coords.append(x1)
        coords.append(y1)
        coords.append(x2)
        coords.append(y2)

    return coords

def get_img_loader(path):
    mean = [0.5432, 0.5432, 0.5432]
    std = [0.3317, 0.3317, 0.3317]
    image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                   ])
    dset = IRDatasetTest(path, transform=image_transforms)
    print(len(dset))
    loader = torch.utils.data.DataLoader(dset, batch_size=16)

    
    return loader

def get_preds(path):
    src = path[:-4]
    ir_img_paths = []
    for file in os.listdir(src):
        if file.endswith(".ir.det.png"):
            ir_img_paths.append(os.path.join(src, file))
            #print(os.path.join(src, file))
    
    # load model
    print(src)
    img_loader = get_img_loader(src)

    model = torch.load("/home/viki/Documents/Informatik/BA/other metrics/ActiveCarModel13.pth", map_location=lambda storage, loc: storage)
    model.eval()

    pred = model(img)
    labels.append(pred)

    print(pred)
    return labels

def display_rgbs():
    # show all rgbs as in a video
    rgb_paths = get_rgbs()
    rgb_paths.sort()
    img = None
    rect = None
    rect1 = None
    rect2 = None
    # ann = None
    for path in rgb_paths:
        im = pl.imread(path)
        fig = pl.figure(1)
        rects = fig.add_subplot(111)
        # annos = fig.add_subplot(111)
        if rect or rect1 or rect2 is not None:
            try:
                rect.remove()
                rect1.remove()
                rect2.remove()
            except:
                pass
        # if ann is not None:
        #     try:
        #         ann.remove()
        #         pass
        #     except:
        #         pass
        coords = get_coords(path)
        #labels = get_preds(path)
        if img is None:
            img = pl.imshow(im)
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            width = x2 - x1
            height = y2 - y1
            rect = pl.Rectangle(xy=(x1, y1), width=width, height=height, color="red", linewidth=2, fill=False)
            # ann = pl.annotate(labels[0], (x1, y1), color="red")
            # annos.add_artist(ann)
            rects.add_artist(rect)
        else:
            img.set_data(im)
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                width = x2 - x1
                height = y2 - y1
                rect = pl.Rectangle(xy=(x1, y1), width=width, height=height, color="red", linewidth=2, fill=False)
                rects.add_artist(rect)
        
        pl.axis('off')
        pl.pause(10)
        pl.draw()

display_rgbs()