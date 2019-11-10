import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import cv2
import sys
import os
from PIL import Image

src_dir = '/home/viki/Documents/Informatik/BA/'
data_dir = 'drive_day_2019_08_21_16_14_06/test2'
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('activemodel12.pth')
model.eval()

def predict_image(image):
    # applies model on PIL image for prediction
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()

    return index

def get_files_list(dir_name):
    # create a list of file and sub directories names in the given directory 
    file_list = os.listdir(dir_name)
    all_files = list()
    for entry in file_list:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_files_list(full_path)
        else:
            all_files.append(full_path)
                
    return all_files

def get_random_images(num=1): # 1
    # returns *num* random image tensor(s) with label(s) and index (indices)
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    images, labels = next(iter(loader))

    return images, labels, idx[0]

    
def normalize(scale_min, ir_path):
    # normalizes an image by image path and returns cv2
    min, max = scale_min, 24000
    im = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
    im = im.astype(np.uint16)
    im = (im.astype(np.float32) - min) / (max - min)
    im = np.clip(im, 0, 1)
    im = (im * 255).astype(np.uint8)
    im_cv = cv2.applyColorMap(im, cv2.COLORMAP_JET)

    return im_cv

def get_im_path(idx):
    # returns path of image to be predicted
    list_of_files = get_files_list(data_dir)
    filename = list_of_files[idx]
    splits = filename.split('/')[2:]
    ir_image_name = f"{src_dir}{data_dir}/{splits[0]}/{splits[1]}"

    return ir_image_name


image, label, idx = get_random_images()
im_to_predict = get_im_path(idx)

image_cv = normalize(22500, im_to_predict)
image_pil = Image.fromarray(np.uint8(image_cv*255))
prediction = predict_image(image_pil)
result = int(label[0]) == prediction

true_state = im_to_predict.split('/')[8]  # active or inactive
if result == 0:
    print(f'Predicted: {prediction}')
    print(f'True state: {true_state}')
    print('Wrong prediction')
    title = f"wrong prediction: is {true_state}"
elif result == 1:
    print(f'Predicted: {prediction}')
    print(f'True state: {true_state}')
    title = f"true prediction: {true_state}"

cv2.imshow(title, image_cv)
key = cv2.waitKey()
if key == 27:
    sys.exit()
