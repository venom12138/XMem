import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
sys.path.append('..')
from model.mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
from copy import deepcopy


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

config_file = '../saves/seg_twohands_ccda/seg_twohands_ccda.py'
checkpoint_file = '../saves/seg_twohands_ccda/best_mIoU_iter_56000.pth'
model = init_segmentor(config_file, checkpoint_file, device=f'cuda')


plt.figure()
for i in range(1446, 2028):
    flow_idx = int(np.ceil((i-3)/2))
    flowu = np.array(Image.open(f"/home/venom/projects/XMem/val_data/P28/flow_frames/P28_19/P28_19_4/u/frame_{str(flow_idx).zfill(10)}.jpg").convert('P'))
    flowv = np.array(Image.open(f"/home/venom/projects/XMem/val_data/P28/flow_frames/P28_19/P28_19_4/v/frame_{str(flow_idx).zfill(10)}.jpg").convert('P'))
    
    # img = np.array(Image.open(f'/home/venom/projects/XMem/val_data/P01/rgb_frames/P01_14/P01_14_90/frame_00000{i}.jpg'))
    # hand_mask = inference_segmentor(model, np.array([img]))[0]
    
    # positions_to_zero = np.where(hand_mask)
    # hand_mask_num = len(positions_to_zero[0])
    # total_num = hand_mask.shape[0]*hand_mask.shape[1]
    # flowu_copy = deepcopy(flowu)
    # # print(torch.sum(flowu_copy) / total_num)
    # flowu_copy[positions_to_zero] = 0
    
    # flowv_copy = deepcopy(flowv)
    # # print(torch.sum(flowv_copy) / total_num)
    # flowv_copy[positions_to_zero] = 0
    
    # target_mean = [np.sum(flowu_copy)/(total_num-hand_mask_num), np.sum(flowv_copy)/(total_num-hand_mask_num)]
    # flowu[positions_to_zero] = target_mean[0]
    # flowv[positions_to_zero] = target_mean[1]
    
    flow = [flowu, flowv]
    flow = np.stack(flow).transpose(1,2,0)
    img = flow_to_image(flow)
    plt.imshow(img)
    # plt.show()
    plt.pause(0.1)
    plt.cla()
    print(i)