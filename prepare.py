import os
import sys
import shutil
import cv2
from PIL import Image, ImageTk
import numpy as np

def sort_data():
    '''
    copy all data in folders from the fl_rgb folder to inactive and active folders
    '''
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_20_06_52/fl_rgb"
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
        

def normalize(scale_min, ir_path):
    # normlizes an image by path, returns normalized as cv2 format
    min, max = scale_min, 24000
    im = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
    im = im.astype(np.uint16)
    im = (im.astype(np.float32) - min) / (max - min)
    im = np.clip(im, 0, 1)
    im = (im * 255).astype(np.uint8)
    im_cv = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    im_cv = cv2.bitwise_not(im_cv) # reverse colormap

    return im_cv

def show_data():
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/test2/active"
    sub_dirs = [x[0] for x in os.walk(src)]

    for folder in sub_dirs:
        for file in os.listdir(folder):
            if file.endswith('.png'):
                filename = os.path.join(folder, file)
                image_cv = normalize(22500, filename)
                cv2.imshow('window', image_cv)
                key = cv2.waitKey()
                if key == 27:
                    cv2.destroyAllWindows()

def rand_data(subset_size=0.2):
    '''
    Move randomized data of size subset_size (%) from src to dst
    '''
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/fr_ir"
    dst = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06_val/fr_ir"
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
    print(len(sub_set))
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



if __name__ == "__main__":
    sort_data()
    #show_data()
    #rand_data()
    #rename_paths()
    #delete_false_dets()

