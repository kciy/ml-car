import thermal_loader
import torch
import numpy as np

from models import *
from utils import *
from save_to_folder import *
import os
import time
import datetime
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable

loader = thermal_loader.ThermalDataLoader(
    "../drive_day_2019_10_10_17_42_32/paths", load_aligned_ir=True)
    # "../drive_day_2019_08_21_16_14_06/paths", load_aligned_ir=True)
train_loader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True, drop_last=True)
config_path = 'config/yolov3-kitti.cfg'
weights_path = 'weights/yolov3-kitti.weights'
class_path = 'data/coco.names'
img_size = 416
conf_thres = 0.85
nms_thres = 0.4
# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_darknet_weights(weights_path)
# model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor


def detectImage(img):
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
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = image_tensor
    # input_img = Variable(image_tensor.type(Tensor))

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, conf_thres, nms_thres)
    return detections[0]


def detect(rgb_path, offset_rel=0.4):
    '''
    Detection of cars in given rgb image path.

    rgb_path : string
        Path to rgb
    offset_rel: float (optional)
        Percentage by which bounding box is enlarged
    '''
    # load image and get detections
    prev_time = time.time()
    img = Image.open(rgb_path)
    detections = detectImage(img)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    #print('Inference Time: %s' % (inference_time))

    img = np.array(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        nr = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if classes[int(cls_pred)] == "car":
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

                # crop image, save image and save txt file
                saveCropped(rgb_path, nr, x1, y1, x2, y2, cls_conf)
                nr += 1

for nr, (out_dict) in enumerate(train_loader):
    rgb_fl = out_dict['rgb_fl']
    rgb_fr = out_dict['rgb_fr']
    ir_fl = out_dict['ir_fl']
    ir_fr = out_dict['ir_fr']
    paths_left = out_dict['paths_left']
    #paths_right = out_dict['paths_right']
    org_left = out_dict['org_left']

    for i in range(len(paths_left)):
        rgb_path_left = paths_left[i][0]
        #rgb_path_right = paths_right[i][0]
        detect(rgb_path_left)
        #detect(rgb_path_right)
