import cv2
import numpy as np
import glob
import os
import time
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow.计算稠密光流"""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -20,20) #default values are +20 and -20
    assert flow.dtype == np.float32
    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow

def cal_for_frames(video_path, save_path):
    #计算flow
    frames = glob.glob(os.path.join(video_path, '*.jpg'))
    frames.sort() #排序
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if not os.path.isdir(f'{save_path}/u'):
        os.makedirs(f'{save_path}/u')
    if not os.path.isdir(f'{save_path}/v'):
        os.makedirs(f'{save_path}/v')
    
    for i, frame_curr in enumerate(frames):
        if i < 7024:
            continue
        elif i == 7024:
            prev = cv2.imread(frame_curr)
            prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)   
            continue
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) #RGB转换为GRAY灰度图
        tmp_flow = compute_TVL1(prev, curr)
        prev = deepcopy(curr)
        u_save_path = os.path.join(save_path, 'u', frame_curr.split('/')[-1])
        v_save_path = os.path.join(save_path, 'v', frame_curr.split('/')[-1])
        
        # print(type(tmp_flow), tmp_flow.shape)
        cv2.imwrite(u_save_path, tmp_flow[:, :, 0])
        cv2.imwrite(v_save_path, tmp_flow[:, :, 1])
        print(u_save_path)

# video_path = '/cluster/home2/yjw/venom/XMem/data/P01/rgb_frames/P01_01'
# save_path = '/cluster/home2/yjw/venom/XMem/data/P01/myflow_frames/P01_01/37'
# if not os.path.isdir(save_path):
#     os.makedirs(save_path)
# cal_for_frames(video_path, save_path)

video_path = '/home/venom/projects/XMem/val_data/P01/rgb_frames/P01_11/P01_11_9'

#计算flow
frames = glob.glob(os.path.join(video_path, '*.jpg'))
frames.sort() #排序
prev = cv2.imread(frames[0])
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

for i, frame_curr in enumerate(frames[1:]):
    curr = cv2.imread(frame_curr)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) #RGB转换为GRAY灰度图
    tmp_flow_forward = compute_TVL1(prev, curr)
    tmp_flow_backward = compute_TVL1(prev, curr)
    prev = deepcopy(curr)
    
    # sum_of_flow = tmp_flow_forward + tmp_flow_backward
    # print(tmp_flow_backward.shape)
    tmp_all = np.array([tmp_flow_forward, tmp_flow_backward]) # = np.concatenate((tmp_flow_forward.unsqueeze(2), tmp_flow_backward.unsqueeze(2)), axis=3)
    reverse_tmp_all = 255 - tmp_all
    sum_of_flow = tmp_all + reverse_tmp_all
    print(tmp_all.shape)
    print((sum_of_flow==255).all())
    print('----------------')