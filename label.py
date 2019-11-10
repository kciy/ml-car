import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import sys
import tkinter as tk


class ImageClassifier(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.window = parent
        self.window.wm_title("Classify image")
        self.counter = 0
        
        self.src = "/home/viki/Documents/Informatik/BA/drive_day_2019_10_10_17_42_32/fl_rgb/"
        # self.src = "/home/viki/Documents/Informatik/BA/drive_day_2019_08_21_16_14_06/fl_rgb/"
        self.list_images = []
        for (dirpath, dirnames, filenames) in os.walk(self.src):
            for filename in filenames:
                if filename.endswith('.det.png') and not filename.endswith('.ir.det.png'):
                    self.list_images.append([dirpath, filename])

        self.frame1 = tk.Frame(self.window, width=500, height=400, bd=2)
        self.frame1.grid(row=1, column=0)
        self.frame2 = tk.Frame(self.window, width=500, height=400, bd=1)
        self.frame2.grid(row=1, column=1)
        self.frame3 = tk.Frame(self.window, width=500, height=400, bd=2)
        self.frame3.grid(row=1, column=2)
        self.frame4 = tk.Frame(self.window, width=1000, height=400, bd=1)
        self.frame4.grid(row=2, column=0, columnspan=3)

        self.cv1 = tk.Canvas(self.frame1, height=390, width=490, background="white", bd=1, relief=tk.SUNKEN)
        self.cv1.grid(row=2, column=0)
        self.cv2 = tk.Canvas(self.frame2, height=390, width=490, background="white", bd=2, relief=tk.SUNKEN)
        self.cv2.grid(row=2, column=1)
        self.cv3 = tk.Canvas(self.frame3, height=390, width=490, background="white", bd=1, relief=tk.SUNKEN)
        self.cv3.grid(row=2, column=2)
        self.cv4 = tk.Canvas(self.frame4, height=390, width=1490, background="white", bd=2, relief=tk.SUNKEN)
        self.cv4.grid(row=3, column=0, columnspan=3)

        self.max_count = len(self.list_images) - 1
        self.switch = True
        self.overwrite = tk.BooleanVar(self.window, False)
        self.overwrite = False
        self.scale = tk.IntVar()
        self.scaleMax = tk.IntVar()
        self.textVar = tk.StringVar()
        self.countVar = tk.StringVar()
        self.labelVar = tk.StringVar()

        label0Button = tk.Button(self.window, text="0: inactive", height=2, width=8, command=lambda: self.classify(0))
        label0Button.grid(row=1, column=3, padx=2, pady=2, sticky=tk.N)
        label1Button = tk.Button(self.window, text="1: active", height=2, width=8, command=lambda: self.classify(1))
        label1Button.grid(row=1, column=4, padx=2, pady=2, sticky=tk.N)
        nextButton = tk.Button(self.window, text="next >>", height=2, width=8, command=self.next_image)
        nextButton.grid(row=0, column=4, padx=2, pady=2)
        prevButton = tk.Button(self.window, text="<< prev", height=2, width=8, command=self.prev_image)
        prevButton.grid(row=0, column=3, padx=2, pady=2)
        overWrCheckbutton = tk.Checkbutton(self.window, text="overwrite", variable=self.overwrite, command=lambda: self.toggle_overwrite())
        overWrCheckbutton.grid(row=0, column=2, padx=2, pady=2, sticky=tk.E)
        normScale = tk.Scale(self.window, from_=21000, to_=23999, length=380, orient=tk.VERTICAL, variable=self.scale, command=lambda v: self.change_overlay(scale=True))
        normScale.grid(row=2, column=4, padx=2, pady=2)
        normMaxScale = tk.Scale(self.window, from_=24000, to_=27000, length=380, orient=tk.VERTICAL, variable=self.scaleMax, command=lambda v: self.change_overlay(scale=True))
        normMaxScale.grid(row=2, column=5, padx=2, pady=2)
        repeatButton = tk.Button(self.window, text="repeat", height=2, width=8, command=lambda: self.change_photo())
        repeatButton.grid(row=2, column=3, padx=2, pady=2, sticky=tk.N)
        falseButton = tk.Button(self.window, text="false detection", height=2, width=16, command=lambda: self.false_detection())
        falseButton.grid(row=1, column=3, columnspan=2, padx=2, pady=2)
        imgLabel = tk.Label(self.window, textvariable=self.textVar, font="Helvetica 12 bold")
        imgLabel.grid(row=0, column=0, padx=2, pady=2)
        countLabel = tk.Label(self.window, textvariable=self.countVar, font="Helvetica 12 bold")
        countLabel.grid(row=0, column=1, padx=2, pady=2)
        labelLabel = tk.Label(self.window, textvariable=self.labelVar, font="Helvetica 12 bold")
        labelLabel.grid(row=0, column=2, padx=2, pady=2)

        self.update_label()
        self.next_image()

        # by capital letters
        self.window.bind('<Escape>', self.close)
        self.window.bind('0', lambda v: self.classify(0))
        self.window.bind('1', lambda v: self.classify(1))
        self.window.bind('r', lambda v: self.change_photo()) # repeat
        self.window.bind('f', lambda v: self.false_detection())

        # arrow keys
        self.window.bind('<Right>', lambda v: self.next_image())
        self.window.bind('<Left>', lambda v: self.prev_image())
        self.window.bind('<Up>', lambda v: self.classify(1))
        self.window.bind('<Down>', lambda v: self.classify(0))

        # wasd
        self.window.bind('w', lambda v: self.classify(1))
        self.window.bind('a', lambda v: self.prev_image())
        self.window.bind('s', lambda v: self.classify(0))
        self.window.bind('d', lambda v: self.next_image())
    
    def classify(self, label):
        path = "{}/{}".format(self.list_images[self.counter-2][0], self.list_images[self.counter-2][1])
        txt_path = path.replace('png', 'txt')
        length = 0
        for el in open(txt_path, "r").read().split():
            length += 1
        if length == 6:
            with open(txt_path, "a") as file:
                file.write(str(label))
                self.labelVar.set(f'wrote label {label} to text file')
                return
        elif length == 7:
            if self.overwrite is False:
                self.labelVar.set(f'image already labelled')
                return
            with open(txt_path, "rb+") as file:
                file.seek(-1, os.SEEK_END)
                file.truncate()
            with open(txt_path, "a") as file:
                file.write(str(label))
                self.labelVar.set(f'changed label to {label}')
        else:
            self.labelVar.set(f'length of txt is neither 6 nor 7: {length} elements')

    def close(self, event):
        sys.exit()
    
    def update_label(self, clear=False):
        self.textVar.set(self.list_images[self.counter-1][1])
        count = self.counter
        if count < 0:
            count = len(self.list_images) + self.counter
        self.countVar.set('{} out of {}'.format(str(count+1), len(self.list_images)))
    
    def false_detection(self):
        dir_name = self.list_images[self.counter-1][0].rsplit("/", 1)[0]
        txt_save_dir = f"{dir_name}/false_detected.txt"
        path = "{}/{}".format(self.list_images[self.counter-2][0], self.list_images[self.counter-2][1])
        self.labelVar.set('false detection registered')
        with open(txt_save_dir, "a") as txt_file:
            print(f"{path}", file=txt_file) # saves rgb path

    def change_overlay(self, scale=False):
        if scale is True:
            self.build_overlay()
            path = "{}/{}".format(self.list_images[self.counter-2][0], self.list_images[self.counter-2][1])
        else:
            path = "{}/{}".format(self.list_images[self.counter-1][0], self.list_images[self.counter-1][1])
        ir_path = path.replace('.det.png', '.ir.det.png')
        
        if not self.counter > self.max_count:
            ir = cv2.imread(ir_path)
            norm_ir_cv = normalize(self.scale.get(), self.scaleMax.get(), ir_path)
            norm_ir_cv = cv2.resize(norm_ir_cv, size(Image.open(path)))
            self.norm_ir_new = Image.fromarray(norm_ir_cv)
            self.ir = ImageTk.PhotoImage(image=self.norm_ir_new)
            
            self.cv3.delete("all")
            self.cv3.create_image(0, 0, anchor="nw", image=self.ir)

    def toggle_overwrite(self):
        self.overwrite = not self.overwrite

    def next_image(self):
        self.update_label()
        self.build_img_list()
        self.labelVar.set('')

        if self.counter > self.max_count:
            self.labelVar.set("No more images")
            self.countVar.set('{} out of {}'.format("-", len(self.list_images)))
            self.textVar.set('')
            self.cv1.delete("all")
            self.cv2.delete("all")
            self.cv3.delete("all")
            self.cv4.delete("all")

        else:
            image = Image.open(self.images[0])
            image = image.resize(size(image, True), Image.ANTIALIAS)
            self.vidphoto = ImageTk.PhotoImage(image)
            self.vidimage = self.cv4.create_image(0, 0, anchor="nw", image=self.vidphoto)
            self.which = 1
            self.window.after(100, self.change_photo)

            path = "{}/{}".format(self.list_images[self.counter-1][0], self.list_images[self.counter-1][1])
            width, height = size(Image.open(path))
            self.next_step(width, height)
    
    def prev_image(self):
        self.counter -= 2
        self.next_image()

    def next_step(self, width, height):
        # displays RGB crop in cv1 and initializes overlay in cv2

        path = "{}/{}".format(self.list_images[self.counter-1][0], self.list_images[self.counter-1][1])
        ir_path = path.replace('.det.png', '.ir.det.png')
        self.change_overlay() # makes self.im_cv, self.img, cv3

        image = Image.open(path)
        # normalized IR as cv2 format
        self.norm_ir_cv2 = normalize(self.scale.get(), self.scaleMax.get(), ir_path)
        self.norm_ir_cv2 = cv2.resize(self.norm_ir_cv2, size(image)) 

        # RGB as cv2 format
        self.im_cv2 = cv2.imread(path)
        self.im_cv2 = cv2.resize(self.im_cv2, size(image)) 

        alpha = 0.7
        beta = 1.0 - alpha
        weighted = cv2.addWeighted((self.im_cv2 * 255).astype(np.uint8), alpha, self.norm_ir_cv2, beta, 0.0)
        self.overlay = ImageTk.PhotoImage(Image.fromarray(weighted))

        # RGB cv1
        self.im = image.resize((int(width), int(height)), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.im)
        
        self.cv1.delete("all")
        self.cv1.create_image(0, 0, anchor="nw", image=self.photo)
        self.cv2.delete("all")
        self.cv2.create_image(0, 0, anchor="nw", image=self.overlay)

        self.counter += 1
    
    def build_overlay(self):
        # cv2: applies addWeighted to the normalized IR image and the RGB on ScaleChange
       
        path = "{}/{}".format(self.list_images[self.counter-2][0], self.list_images[self.counter-2][1])
        ir_path = path.replace('.det.png', '.ir.det.png')
        
        image = Image.open(path)
        # normalized IR as cv2 format
        self.norm_ir_cv = normalize(self.scale.get(), self.scaleMax.get(), ir_path)
        self.norm_ir_cv = cv2.resize(self.norm_ir_cv, size(image))

        # RGB as cv2 format
        self.im_cv_u = cv2.imread(path)
        self.im_cv_u = cv2.resize(self.im_cv_u, size(image))

        alpha = 0.7
        beta = 1.0 - alpha
        weighted = cv2.addWeighted((self.im_cv_u * 255).astype(np.uint8), alpha, self.norm_ir_cv, beta, 0.0)
        self.new_overlay = ImageTk.PhotoImage(Image.fromarray(weighted))

        self.cv2.delete("all")
        self.cv2.create_image(0, 0, anchor="nw", image=self.new_overlay)
    
    def build_img_list(self):
        # makes self.images, a list of 4 images for cv4 (original big RGB images)

        list_dir = os.listdir(self.src)
        sorted_list_dir = sorted(list_dir)
        if not self.counter > self.max_count:
            path = "{}/{}".format(self.list_images[self.counter-1][0], self.list_images[self.counter-1][1])
            orig = os.path.dirname(path) + '.png'
            pre_name = self.list_images[self.counter-1][0].rsplit("/", 1)[0]
            idx = sorted_list_dir.index(orig.split("/")[-1])
            image_list = []
            for i in range(4):
                img_name = "{}/{}".format(pre_name, sorted_list_dir[idx+i])
                image_list.append(img_name)
            self.images = image_list

    def change_photo(self):
        # displays original image and the 3 following big RGBs in cv4
        
        self.switch = True
        if self.which == 0:
            self.image = Image.open(self.images[0])
            self.which = 1
        elif self.which == 1:
            self.image = Image.open(self.images[1])
            self.which = 2
        elif self.which == 2:
            self.image = Image.open(self.images[2])
            self.which = 3
        else:
            self.image = Image.open(self.images[3])
            self.which = 0
            self.switch = False
        
        self.image = self.image.resize(size(self.image, True), Image.ANTIALIAS)
        self.vidphoto_orig = ImageTk.PhotoImage(self.image)
        self.cv4.itemconfig(self.vidimage, image=self.vidphoto_orig)
        if self.switch is True:
            self.window.after(100, self.change_photo)
    
def size(image, large=False):
    # returns (width, height) for a given (small) image

    w = 1490 if large is True else 490
    h = 390
    if (w - image.size[0]) < (h - image.size[1]):
        width = w
        height = width * image.size[1] / image.size[0]
    else:
        height = h
        width = height * image.size[0] / image.size[1]
    return (int(width), int(height))

def normalize(scale_min, scale_max, ir_path):
    # normlizes an image by path, returns normalized as cv2 format

    min, max = scale_min, scale_max
    im = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
    im = im.astype(np.uint16)
    im = (im.astype(np.float32) - min) / (max - min)
    im = np.clip(im, 0, 1)
    im = (im * 255).astype(np.uint8)
    im_cv = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    im_cv = cv2.bitwise_not(im_cv) # reverse colormap

    return im_cv
        

if __name__ == "__main__":
    window = tk.Tk()
    MyApp = ImageClassifier(window)
    tk.mainloop()
