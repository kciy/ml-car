import os
import sys
import shutil
import cv2
from PIL import Image, ImageTk
import numpy as np

def sort_data():
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/fl_rgb"
    dst0 = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/test2/inactive"
    dst1 = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/test2/active/"

    # list_0: 82/661 = 12,41%
    # list_1: 579/661 = 87,59%
    # 29 false detections => 690 folders in total

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
                    if last_char == "0":
                        list_0.append(ir)
                    if last_char == "1":
                        list_1.append(ir)
    print(len(list_0))
    print(len(list_1))
    for f in list_0:
        fooo = f.split("/")[-1]
        #print(f)
        #print(os.path.join(dst0, fooo))
        shutil.copy(f, os.path.join(dst0, fooo))
    for f in list_1:
        foo = f.split("/")[-1]
        shutil.copy(f, os.path.join(dst1, foo))
        

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

def rand_data(size=0.15):
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/fr_ir"
    dst = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_for_test"
    sub_dirs = [x[0] for x in os.walk(src)]
    all_files = []
    for folder in sub_dirs:
        for file in os.listdir(folder):
            filename = os.path.join(folder, file)
            all_files.append(filename)
    len_files = len(all_files)
    indices = list(range(len_files))
    split = int(np.floor(size * len_files))
    np.random.shuffle(indices)

    train_set, test_set = indices[split:], indices[:split]
    for idx in test_set:
        print(all_files[idx])
        shutil.copy(all_files[idx], dst)
        os.remove(all_files[idx])
    print("Done")

def rename_paths():
    src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/paths_original"
    all_files = []
    for file in os.listdir(src):
        filename = os.path.join(src, file)
        all_files.append(filename)
    for txt in all_files:
        fin = open(txt, "rt")
        data = fin.read()
        data = data.replace("vertensj/Documents/robocar_bags/dumped/10_10_19_day", "viki/Documents/Informatik/BA")
        fin.close()

        fin = open(txt, "wt")
        fin.write(data)
        fin.close()
        print("Done")
    
def delete_false_dets():
    src0 = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/fl_rgb/train/active/"
    src1 = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/test2/fl_rgb/train/inactive/"
    src2 = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/fl_rgb/"

    txt = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/fl_rgb/false_detected.txt"

    fin = open(txt, "r")
    lines = fin.readlines()
    i = 0
    for line in lines:
        name = line.split('/')[-1]
        filename0 = src2 + name
        filename0 = filename0[:-11]
        filename0 = filename0 + '.png'
        filename1 = src2 + name
        filename1 = filename1[:-11]
        filename1 = filename1 + '.png'
        print(filename0)
        if os.path.exists(filename0):
            os.remove(filename0)
            print("yep 0")
        elif os.path.exists(filename1):
            os.remove(filename1)
            print("yep 1")
        else:
            print("Nope")
        i += 1
        print(f"Done {i}/{len(lines)+1}")



if __name__ == "__main__":
    #sort_data()
    #show_data()
    #rand_data()
    #rename_paths()
    delete_false_dets()

