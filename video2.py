import os
import pylab as pl
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
import torchvision
import cv2
from torch import nn
from torch import optim
import time
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from IRDataset import IRDatasetTest
from models import *
from utils import *

config_path = 'PyTorch-YOLOv3/config/yolov3.cfg'
weights_path = 'PyTorch-YOLOv3/weights/yolov3.weights'
class_path = 'PyTorch-YOLOv3/data/coco.names'
img_size = 416
conf_thres = 0.65
nms_thres = 0.4
model = Darknet(config_path, img_size=img_size)
model.load_darknet_weights(weights_path)
model.eval()
classes = utils.load_classes(class_path)

global dataset
# dataset = "night"
# dataset = "day"
dataset = "combined"

def get_rgbs():
    # get sequence of rgb full images
    # src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_06_52/fl_rgb" # night
    # src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_16_52/fl_rgb" # night1
    # src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_26_53/fl_rgb" # night2
    # src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_36_52/fl_rgb" # night3
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/fl_rgb" # day1
    # src = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/fl_rgb" # day2
    # src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/fl_rgb" # day3
    folders = [x[0] for x in os.walk(src)]
    folders.sort()
    root = src.replace("fl_rgb", "")
    rgb_names = [root + "fl_rgb/" + x.split("/")[-1] + '.png' for x in folders]
    rgb_names = rgb_names[1:]
    return rgb_names

def detectImage(img):
    # between 280 and 1680px
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh - imw) / 2), 0),
              max(int((imw - imh) / 2), 0), max(int((imh - imw) / 2), 0),
              max(int((imw - imh) / 2), 0)), (128, 128, 128)),
         transforms.ToTensor(),
         ])
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = image_tensor

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, conf_thres, nms_thres)
    img = np.array(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    offset_rel = 0.4
    coords = []
    for det in detections:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in det:
            if det is not None and classes[int(cls_pred)] == "car":
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                offset_w = box_w * offset_rel
                offset_h = box_h * offset_rel
                box_w += offset_w
                box_h += offset_h
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1] - offset_w / 2
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0] - offset_h / 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x1 < 270 or y1 < 270 or x2 > 1670 or y2 > 1670:
                    pass # nada
                else:
                    coords.append([x1, y1, x2, y2])

    return coords

def get_color(img, model):
    img_3chan = cv2.merge((img, img, img))
    img = cv2.resize(img_3chan, (224, 224))
    image_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                   ])
    img = image_transforms(img)
    img = img.unsqueeze(0)
    outputs = model(img)
    outputs = torch.exp(outputs)
    _, preds = torch.max(outputs,1)

    outputs = outputs.detach().numpy()
    label = preds
    label = label.numpy()
    conf = 0.
    if label == 0:
        conf = outputs[0][0]*100
        label = "active {0:0.2f}%".format(conf)
    if label == 1:
        conf = outputs[0][1]*100
        label = "inactive {0:0.2f}%".format(conf)
    color = ""
    if "active" in label:
        if conf > 90:
            color = "#b21f35" # red
        if conf < 90 and conf > 70:
            color = "#ff7435" # orange
        if conf < 70:
            color = "#ffcb35" # yellow
    if "inactive" in label:
        color = "#000000"
    #     if conf > 90:
    #         color = "#0052a5"
    #     if conf < 90 and conf > 70:
    #         color = "#0079e7"
    #     if conf < 70:
    #         color = "#06a9fc"

    return color

def detect_imgs(class_model):
    img_paths = get_rgbs()
    img = pl.imshow(np.empty([650,1920,3], dtype=np.uint8))
    num = 0
    for img_path in img_paths:
        since = time.time()
        fig = pl.figure(1)
        rects = fig.add_subplot(111)
        for i in rects.get_children():
            if "Rectangle" in str(i):
                i.set_visible(False)
        img_det = Image.open(img_path)
        coords = detectImage(img_det)
        img_cv = pl.imread(img_path)
        ir_path = img_path.replace("rgb", "ir_aligned")
        ir = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
        ir = ir.astype(np.uint16)
        ir = (ir.astype(np.float32) - 24000) / (21800 - 24000)
        ir_col = np.clip(ir, 0, 1)
        ir = (ir_col * 255).astype(np.uint8)
        ir = cv2.applyColorMap(ir, cv2.COLORMAP_JET)
        img_cv = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        car_count = 0
        for x1, y1, x2, y2 in coords:
            color = get_color(ir_col[y1:y2, x1:x2], class_model)
            if color != "#000000":
                car_count += 1
                ir_overlay = img_cv.copy()
                ir_overlay[y1:y2, x1:x2] = ir[y1:y2, x1:x2]
                alpha = 0.5
                img_cv = cv2.addWeighted(ir_overlay, alpha, img_cv, 1-alpha, 0, img_cv)
                rects.add_artist(pl.Rectangle(xy=(x1,y1), width=x2-x1, height=y2-y1, color=color, linewidth=2, fill=False))
            else:
                car_count += 1
                rects.add_artist(pl.Rectangle(xy=(x1,y1), width=x2-x1, height=y2-y1, color=color, linewidth=2, fill=False))
        
        time_elapsed = time.time() - since
        print("{:.3f}s for {:.0f} cars".format(time_elapsed % 60, car_count))

        img.set_data(img_cv)

        if num == 0:
            pl.plot(0, 0, "-", c="#b21f35", label="conf > 90%")
            pl.plot(0, 0, "-", c="#ff7435", label="conf > 70%")
            pl.plot(0, 0, "-", c="#ffcb35", label="conf > 50%")
            pl.plot(0, 0, "-", c="#000000", label="inactive")
            # pl.plot(0, 0, "-", c="#0052a5", label="inactive and conf > 90%")
            # pl.plot(0, 0, "-", c="#0079e7", label="inactive and conf > 70%")
            # pl.plot(0, 0, "-", c="#06a9fc", label="inactive and conf > 50%")
        pl.legend(bbox_to_anchor=(1, 1), ncol=1, loc=4, prop={'size': 7})
        num += 1
        val = 2.0 - time_elapsed
        if val < 0:
            val = time_elapsed
        sum = val + time_elapsed
        print(int(sum))
        pl.axis('off')
        pl.pause(val)
        pl.draw()


if __name__ == "__main__":
    # MODELPATH0 = "/home/viki/Documents/Informatik/BA/ActiveCarModel12.pth" # night
    # MODELPATH1 = "/home/viki/Documents/Informatik/BA/ActiveCarModel10.pth" # day
    MODELPATH2 = "/home/viki/Documents/Informatik/BA/ActiveCarModel11.pth" # combined
    if dataset == "night":
        MODELPATH = MODELPATH0
        mean = [0.6968, 0.6968, 0.6968]
        std = [0.2580, 0.2580, 0.2580]
    elif dataset == "day":
        MODELPATH = MODELPATH1
        mean = [0.5094, 0.5094, 0.5094]
        std = [0.3474, 0.3474, 0.3474]
    else:
        MODELPATH = MODELPATH2
        mean = [0.5432, 0.5432, 0.5432]
        std = [0.3317, 0.3317, 0.3317]
    class_model = torchvision.models.resnet18(num_classes=2)
    num_features = class_model.fc.in_features
    class_model.fc = nn.Sequential(nn.Linear(num_features, 512),
                             nn.LeakyReLU(0.3), 
                             nn.Dropout(0.1), 
                             nn.Linear(num_features, 2), 
                             nn.LogSoftmax(dim=1))

    state_dict = torch.load(MODELPATH, map_location="cpu")
    class_model.load_state_dict(state_dict, strict=False)
    class_model.eval()
    detect_imgs(class_model)