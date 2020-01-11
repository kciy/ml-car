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

def normalize():
    '''
    print mean and std for given dataset
    '''
    dataset = datasets.ImageFolder('drive_all_train',
                               transform=transforms.Compose([
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()]))

    loader = DataLoader(dataset,
                        batch_size=10,
                        num_workers=3,
                        shuffle=False)
    mean = 0.
    std = 0.
    for images, _ in loader:
        # batch size (the last batch can have smaller size!)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    print('mean: ', mean)
    print('std: ', std)


def sort_data():
    '''
    copy all data in folders from the fl_rgb folder to inactive and active folders
    '''
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_36_52/fl_rgb"
    dst0 = "/home/viki/Documents/Informatik/BA/drive_all_night/inactive/"
    dst1 = "/home/viki/Documents/Informatik/BA/drive_all_night/active/"

    sub_dirs = [x[0] for x in os.walk(src)]
    list_0 = []
    list_1 = []

    for folder in sub_dirs:
        for file in os.listdir(folder):
            if file.endswith('.txt'):
                filename = os.path.join(folder, file)
                with open(filename, 'rb') as f:
                    last_char = str(f.read())[-2]
                    ir = filename.replace('det.txt', 'ir.det.png')
                    # print(ir)
                    if last_char == "0":
                        list_0.append(ir)
                    if last_char == "1":
                        list_1.append(ir)
    print(len(list_0))
    print(len(list_1))
    for f in list_0:
        file0 = f.split("/")[-1]
        #dst = os.path.join(dst0, file0)
        shutil.copy(f, dst0)
    for f in list_1:
        file1 = f.split("/")[-1]
        #dst = os.path.join(dst1, file1)
        shutil.copy(f, dst1)
        

def normalize_image(scale_min, ir_path):
    # normlizes an image by path, returns normalized as cv2 format
    min, max = 21000, 24200
    min, max = 21800, 24000
    im = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
    im = im.astype(np.uint16)
    im = (im.astype(np.float32) - min) / (max - min)
    im = np.clip(im, 0, 1)
    im = (im * 255).astype(np.uint8)
    im_cv = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    #im_cv = cv2.bitwise_not(im_cv) # reverse colormap

    return im_cv

def false_det(src, filename):
    txt_save_dir = f"{src}/false_detected.txt"
    with open(txt_save_dir, "a") as txt_file:
        print(f"{filename}", file=txt_file)

def show_data():
    # src = "/home/viki/Documents/Informatik/BA/drive_all_day" # active 3138, inactive 607
    # src = "/home/viki/Documents/Informatik/BA/drive_all_night" # active 669, inactive 148
    src = "/home/viki/Documents/Informatik/BA/drive_all" # active 3807, inactive 755 (4562)
    sub_dirs = [x[0] for x in os.walk(src)]
    i = 0
    for folder in sub_dirs:
        print(folder)
        for file in os.listdir(folder):
            if file.endswith('.png'):
                i += 1
                print(i)
                filename = os.path.join(folder, file)
                image_cv = normalize_image(22500, filename)
                image_cv = cv2.resize(image_cv, (224, 224))
                cv2.imshow(folder.rsplit('/')[7], image_cv)
                key = cv2.waitKey()
                if key == 102: # f for false! (also for exit)
                    false_det(src, filename)
                    cv2.destroyAllWindows()

def rand_data(subset_size=0.2):
    '''
    Move randomized data of size subset_size (%) from src to dst
    '''
    src = "/home/viki/Documents/Informatik/BA/drive_all_day_train/active"
    dst = "/home/viki/Documents/Informatik/BA/drive_all_day_test_20/active"
    sub_dirs = [x[0] for x in os.walk(src)]
    all_files = []
    for folder in sub_dirs:
        for file in os.listdir(folder):
            filename = os.path.join(folder, file)
            all_files.append(filename)
    len_files = len(all_files)
    indices = list(range(len_files))
    split = int(np.floor(subset_size * len_files))
    np.random.shuffle(indices)

    train_set, sub_set = indices[split:], indices[:split]
    # print(len(train_set))
    # print(len(sub_set))
    for idx in sub_set:
        print(all_files[idx])
        shutil.copy(all_files[idx], dst)
        os.remove(all_files[idx])
    print("Done")

def rename_paths():
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_52_31/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_18_02_31/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_18_12_31/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_18_22_31/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_19_46_53/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_19_56_52/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_06_52/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_16_52/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_26_53/paths_original"
    #src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_36_52/paths_original"
    src = ""
    
    all_files = []
    for file in os.listdir(src):
        filename = os.path.join(src, file)
        all_files.append(filename)
    for txt in all_files:
        fin = open(txt, "rt")
        data = fin.read()
        # data = data.replace("vertensj/Documents/robocar_bags/dumped/10_10_19_day", "viki/Documents/Informatik/BA")
        data = data.replace("vertensj/Documents/robocar_bags/dumped/10_10_19_night", "viki/Documents/Informatik/BA")
        fin.close()

        fin = open(txt, "wt")
        fin.write(data)
        fin.close()
        print("Done")
    
def delete_false_dets():
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_26_53/fl_rgb/"
    txt = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_26_53/fl_rgb/false_detected.txt"

    fin = open(txt, "r")
    lines = fin.readlines()
    i = 0
    for line in lines:
        name = line.split('/')[-1]
        # print(name)
        filename = name[:-9]
        folder = name[:-11]
        file = src + folder + '/' + filename + '.det.png'
        filenametxt = src + folder + '/' + filename + '.det.txt'
        irfile = src + folder + '/' + filename + '.ir.det.png'
        
        # print(file)
        # print(filenametxt)
        # print(irfile)
        
        if os.path.exists(file):
            os.remove(file)
            # print("Yep png")
        else:
            print("Nope png")
        if os.path.exists(filenametxt):
            os.remove(filenametxt)
            # print("Yep txt")
        else:
            print("Nope txt")
        if os.path.exists(irfile):
            os.remove(irfile)
            # print("Yep ir")
        else:
            print("Nope ir")
        
        i += 1
        print(f"Done {i}/{len(lines)}")


def plot():
    # fn = [11, 11, 11, 11, 11, 10, 9, 8, 7, 4,4,2,5,4,4,3,3,1,2,1,2,1,3,2,2,4,2,2,1,2,1,2,1,1,1,3,1,1,1,1,4,1,2,1,2,1,1,1,1,2]
    # fp = [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,2,1,1,0,1,2,1,1,0,1,1,1,2,1,1,1,0,1,0,1,0,1,0,1,0,1]
    precRec = []
    '''
    plt.figure(6)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(precRec)
    plt.plot(fn)
    plt.grid()
    plt.legend(frameon=False)
    '''
    import pandas as pd

    recall = [0, 0.1, 0.2, 0.4, 0.8]
    precision = [1, 0.1, 0.95, 0.2, 0.7]

    df = pd.DataFrame({'x': recall, 'y1': precision})
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot('x', 'y1', data=df, marker="", linewidth=1.15)

    plt.show()


def show_img_from_file():
    image_path = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_52_31/fl_ir/fl_ir_1570722756_5403258402.png"
    img = normalize_image(21000,image_path)
    cv2.imshow('window', img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

        
def avg_array():
    txt0 = "/home/viki/Documents/Informatik/von ya/6dwn3xdr/f1_history.txt" # night
    txt1 = "/home/viki/Documents/Informatik/von ya/1sv62nza/f1_history.txt" # day
    txt2 = "/home/viki/Documents/Informatik/von ya/wfr3nvou/f1_history.txt" # combined

    fin = open(txt0, "r")
    lines = fin.readlines()
    avg = 0
    best = 0
    for i in lines:
        avg += float(i)
        if float(i) > best:
            best = float(i)
    print("best f1 night: ", best)
    print("avg f1 night: ", avg/len(lines))
    print('-------')

    fin = open(txt1, "r")
    lines = fin.readlines()
    avg = 0
    best = 0
    for i in lines:
        avg += float(i)
        if float(i) > best:
            best = float(i)
    print("best f1 day: ", best)
    print("avg f1 day: ", avg/len(lines))
    print('-------')

    fin = open(txt2, "r")
    lines = fin.readlines()
    avg = 0
    best = 0
    for i in lines:
        avg += float(i)
        if float(i) > best:
            best = float(i)
    print("best f1 combined: ", best)
    print("avg f1 combined: ", avg/len(lines))

def sort_out():
    srctxt = "/home/viki/Documents/Informatik/BA/drive_all/false_detected.txt"
    a = 0
    d = 0
    n = 0

    lines = open(srctxt, 'r').readlines()
    if True:
        for img in lines:
            img = img.rsplit('/')[6] + '/' + \
                  img.rsplit('/')[7] + '/' + \
                  img.rsplit('/')[8]
            img_day = img.rsplit('/')[0] + '_day/' + \
                      img.rsplit('/')[1] + '/' + \
                      img.rsplit('/')[2]
            img_night = img.rsplit('/')[0] + '_night/' + \
                      img.rsplit('/')[1] + '/' + \
                      img.rsplit('/')[2]
            img = img[:-1]
            img_day = img_day[:-1]
            img_night = img_night[:-1]
            if os.path.isfile(img):
                os.remove(img)
                a += 1
                print(f"all: {a}, day: {d}, night: {n}")
            if os.path.isfile(img_day):
                os.remove(img_day)
                d += 1
                print(f"all: {a}, day: {d}, night: {n}")
            if os.path.isfile(img_night):
                os.remove(img_night)
                n += 1
                print(f"all: {a}, day: {d}, night: {n}")


if __name__ == "__main__":
    # normalize()
    # sort_data()
    # show_data()
    rand_data()
    # rename_paths()
    # delete_false_dets()
    # plot()
    # show_img_from_file()
    # avg_array()
    # sort_out()

    '''
    copy files from pearl:
    scp -o ProxyJump=schwarzv@aislogin.informatik.uni-freiburg.de schwarzv@pearl2:/home/schwarzv/... /home/viki/...

    scp -o ProxyJump=schwarzv@aislogin.informatik.uni-freiburg.de /home/viki/Documents/Informatik/BA/drive_all_day_train/* schwarzv@pearl2:/home/schwarzv/Documents/drive_all_day_train
    '''