import thermal_loader
import cv2
import torch
import numpy as np


def normalize(disp_cv, range_min, range_max):
    disp_cv = disp_cv.astype(np.uint16)
    print('range_min = {}'.format(range_min))
    print('range_max = {}'.format(range_max))
    range_min *= 86
    range_max *= 105
    print('range_min = {}'.format(range_min))
    print('range_max = {}'.format(range_max))
    print('-------------')
    
    disp_cv = (disp_cv.astype(np.float32) - range_min) / (range_max - range_min)
    disp_cv = np.clip(disp_cv, 0, 1)
    disp_cv = (disp_cv * 255).astype(np.uint8)
    disp_cv_color = cv2.applyColorMap(disp_cv, cv2.COLORMAP_JET)

    return disp_cv_color

def visIr(data, name, min_value, max_value, min_val=20000):
    disp_cv = data.cpu().data.numpy().squeeze()
    print(disp_cv.shape)
    disp_cv = normalize(disp_cv, min_value, max_value)
    print(disp_cv.shape)
    #cv2.imshow(name, disp_cv)
    return disp_cv

def visImage3Chan(data, name):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    # cv2.imshow(name, cv)
    return cv

def overlayThermal(rgb, thermal, nr):
    alpha = 0.7  # 0.4
    beta = 1.0 - alpha
    weighted = cv2.addWeighted((rgb * 255).astype(np.uint8), alpha, thermal, beta, 0.0)
    cv2.imshow('Overlayed', weighted)

def nothing(x):
    pass


cv2.namedWindow('Overlayed')
loader = thermal_loader.ThermalDataLoader(
    'drive_day_2019_08_21_16_14_06/paths', load_aligned_ir=True)
train_loader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True, drop_last=True)

for nr, (out_dict) in enumerate(train_loader):
    rgb_fl = out_dict['rgb_fl']
    rgb_fr = out_dict['rgb_fr']
    ir_fl = out_dict['ir_fl']
    ir_fr = out_dict['ir_fr']
    paths_left = out_dict['paths_left']
    org_left = out_dict['org_left']

    rgb = visImage3Chan(rgb_fl[0], 'RGB Left')

    data = ir_fl[0]
    min_val = 20000
    disp_cv = data.cpu().data.numpy().squeeze()

    min_value = 0  # 20000
    max_value = 255  # 27000
    disp_cv = normalize(disp_cv, min_value, max_value)

    cv2.createTrackbar('Min', 'Overlayed', min_value, max_value, nothing)
    cv2.createTrackbar('Max', 'Overlayed', max_value, max_value, nothing)

    next_image = False
    while next_image is False:
        min_value = cv2.getTrackbarPos('Min', 'Overlayed')
        max_value = np.max([min_value, cv2.getTrackbarPos('Max', 'Overlayed')])

        if min_value == max_value:
            cv2.setTrackbarPos('Max', 'Overlayed', min_value)

        if min_value < 233:
            cv2.setTrackbarPos('Min', 'Overlayed', 233)

        thermal = visIr(ir_fl[0], 'IR Left', min_value, max_value)
        overlayThermal(rgb, thermal, nr)

        key = cv2.waitKey()
        if key == 32 or key == 13:  # space or enter for next image
            next_image = True
        elif key == 27:  # ESC to exit
            import sys
            sys.exit()
        