import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import cv2


def saveCropped(rgb_path, nr, x1, y1, x2, y2, cls_conf=None, kitti=False, ir_path_real=""):
    '''
    Saves cropped rgb, ir and txt to one folder named by rgb image path once
    per detected car above threshold for class confidence.
    Txt file saves: rgb_path, class_confidence, (x1, y1, x2, y2) coordinates
    of bounding box.

    rgb_path : string
        File name of rgb
    nr : int
        Xth detection to differ savings in one folder
    x1, y1, x2, y2 : int
        Coordinates of bounding box of detected car
    cls_conf : float (optional)
        Class confidence of detection
    kitti : boolean (optional)
        True if kitti weights and data are used
    '''
    base = os.path.basename(rgb_path)
    ir_path = rgb_path.replace('rgb', 'ir_aligned')
    # ir_path = rgb_path.replace('rgb', 'ir')
    img_name = os.path.splitext(base)[0]
    img_dir = os.path.dirname(rgb_path) + '/' + img_name
    img_dir_ir = os.path.dirname(rgb_path) + '/' + img_name
    
    rgb_save_dir = '{0}/{1}_{2}.det.png'.format(img_dir, img_name, str(nr))
    ir_save_dir = '{0}/{1}_{2}.ir.det.png'.format(img_dir_ir, img_name, str(nr))
    txt_save_dir = '{0}/{1}_{2}.det.txt'.format(img_dir, img_name, str(nr))

    # only save if both RGB and IR exist and mask
    if not os.path.isfile(rgb_path):
        #print(f'{rgb_path} doesnt exist')
        return
    if not os.path.isfile(ir_path):
        # print(f'{ir_path} doesnt exist')
        return
    
    dpi = 200.0

    # crop and show RGB
    rgb = Image.open(rgb_path)
    rgb = np.array(rgb)
    cropped_rgb = rgb[y1:y2, x1:x2]
    height, width, nbands = cropped_rgb.shape
    figsize = (width + 1) / dpi, (height + 1) / dpi

    fig = plt.figure(rgb_path, figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(cropped_rgb)

    # crop and show IR
    ir = cv2.imread(ir_path, cv2.IMREAD_ANYDEPTH)
    cropped_ir = ir[y1:y2, x1:x2]
    if cropped_ir[0].size == 0:
        # print('zero size')
        return
    # ir_new = cv2.imread(ir_path_real, cv2.IMREAD_ANYDEPTH)
    # cropped_ir_new = ir_new[y1:y2, x1:x2]
    # if cropped_ir_new[0].size == 0:
        # print('zero size)
        # return
    
    #if not irCoveredMask(cropped_ir):
    #    return
    # if not irCoveredMask(cropped_ir_new):
        # return
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        # print('created new directory')
    
    cv2.imwrite(ir_save_dir, cropped_ir)
    # cv2.imwrite(ir_save_dir_new, cropped_ir_new)

    # save rgb + txt
    fig.savefig(rgb_save_dir, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    cls_conf = "{0:,.4f}".format(float(cls_conf))
    with open(txt_save_dir, "w") as txt_file:
        print(f"{rgb_path} {float(cls_conf)} {x1} {y1} {x2} {y2}", file=txt_file)
    print(f'saved one more txt')


def irCoveredMask(ir, min_val=20000, mask=100):
    '''
    Tests if the given infrared image is fully recorded or has more than <mask>
    white pixels.
    Returns true if IR is fully recorded, false otherwise.

    ir : array
        Cropped image
    min_val : int (optional)
        Minimum threshold to test against
    mask : int (optional)
        Count threshold for amount of pixels under min_val
    '''
    off_px = 0
    for col in range(ir.shape[0]):
        for px in range(ir.shape[1]):
            if ir[col][px] < min_val/255:
                off_px += 1
    if off_px > mask:
        return False
    else:
        return True
