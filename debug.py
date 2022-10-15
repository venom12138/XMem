import cv2
import numpy as np
from glob import glob
import os
import time
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image
import matplotlib.pyplot as plt
import seaborn
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
import random
from util.exp_handler import *
import sys
from argparse import ArgumentParser
from pathlib import Path
import torchvision.transforms as transforms

# im_normalization = transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
# final_im_transform = transforms.Compose([
#             transforms.ToTensor(),
#             # im_normalization,
#         ])

# rgb_path = '/home/venom/projects/XMem/val_data/P01/rgb_frames/P01_11/P01_11_9/frame_0000002646.jpg'
# vid_flowu_path = '/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/u/frame_0000001322.jpg'
# vid_flowv_path = '/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/v/frame_0000001322.jpg'

# flow_dir = '/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/u'
# for flow_u in glob(flow_dir + '/*.jpg'):
#     flow_u = Image.open(flow_u).convert('P')
#     # print(torch.max(torch.from_numpy(np.array(flow_u))))
#     flow_u = final_im_transform(flow_u)
#     flow_u = flow_u - torch.mean(flow_u)
#     print(flow_u.shape)
#     flow = torch.cat([flow_u, flow_u], dim=0)
#     print(flow.shape)
    # print(torch.min(flow_u))

# this_flowu = Image.open(vid_flowu_path).convert('P')
# this_flowv = Image.open(vid_flowv_path).convert('P')
# this_img = Image.open(rgb_path).convert('RGB')

# print(final_im_transform(this_img).shape)
# print(np.array(this_flowu).shape)
# print(torch.from_numpy(np.array(this_flowu)).shape)
# print(torch.stack([torch.from_numpy(np.array(this_flowu)), torch.from_numpy(np.array(this_flowv))], dim=0).shape)
# print(transforms.ToTensor()(this_flowu).shape)
# print(np.savetxt('testu.out', transforms.ToTensor()(this_flowu)[0]))
# print(np.savetxt('testv.out', transforms.ToTensor()(this_flowv)))
# np.savetxt('testrgb.out', np.array(final_im_transform(this_img)[0]))
# print(this_flowu.size)
# aggregate flow
# agg_u_frames = []
# agg_v_frames = []
# flow_name = 'frame_0000001327.jpg'
# vid_flow_path = '/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9'
# u_path = os.path.join(vid_flow_path, 'u')
# v_path = os.path.join(vid_flow_path, 'v')
# all_u_jpgs = sorted(glob(f'{u_path}/*.jpg'))
# all_v_jpgs = sorted(glob(f'{v_path}/*.jpg'))
# assert len(all_u_jpgs) > 5 and len(all_v_jpgs) > 5
# u_idx = all_u_jpgs.index(os.path.join(vid_flow_path, 'u', flow_name))
# v_idx = all_v_jpgs.index(os.path.join(vid_flow_path, 'v', flow_name))
# if u_idx == 0 or u_idx == 1:
#     agg_u_frames = all_u_jpgs[:5]
# elif u_idx == len(all_u_jpgs) - 1 or u_idx == len(all_u_jpgs) - 2:
#     agg_u_frames = all_u_jpgs[-5:]
# else:
#     agg_u_frames = all_u_jpgs[u_idx-2:u_idx+3]

# if v_idx == 0 or v_idx == 1:
#     agg_v_frames = all_v_jpgs[:5]
# elif v_idx == len(all_v_jpgs) - 1 or v_idx == len(all_v_jpgs) - 2:
#     agg_v_frames = all_v_jpgs[-5:]
# else:
#     agg_v_frames = all_v_jpgs[v_idx-2:v_idx+3]

# print(f'agg_u_frames: {agg_u_frames}')
# print(f'agg_v_frames: {agg_v_frames}')

# this_flow = None
# # process all flow frames
# for tmp_idx in range(len(agg_u_frames)):
    
#     this_flowu = Image.open(agg_u_frames[tmp_idx]).convert('P')

#     this_flowv = Image.open(agg_v_frames[tmp_idx]).convert('P')
    
#     # 将0-255的像素值映射到0到1之间并中心化
#     this_flowu = transforms.ToTensor()(this_flowu)
#     this_flowv = transforms.ToTensor()(this_flowv)
#     this_flowu = this_flowu - torch.mean(this_flowu)
#     this_flowv = this_flowv - torch.mean(this_flowv)
    
#     # this_flow 最后的shape是2*L x H x W
#     if this_flow == None:
#         this_flow = torch.cat([this_flowu, this_flowv], dim=0)
#     else:
#         this_flow = torch.cat([this_flow, this_flowu, this_flowv], dim=0)
    
# # this_flow = torch.stack(this_flow, 0)
# print(f'this_flow:{this_flow.shape}')
# print(f'this_flow:{torch.std(this_flow)}')

mask = Image.open('/home/venom/projects/XMem_evaluation/results/semi-supervised/osvos/bike-packing/00001.png').convert('P')
print(np.unique(mask))
# print(np.unique(mask))