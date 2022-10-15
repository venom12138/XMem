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

im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            # im_normalization,
        ])

rgb_path = '/home/venom/projects/XMem/val_data/P01/rgb_frames/P01_11/P01_11_9/frame_0000002646.jpg'
vid_flowu_path = '/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/u/frame_0000001322.jpg'
vid_flowv_path = '/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/v/frame_0000001322.jpg'

flow_dir = '/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/u'
for flow_u in glob(flow_dir + '/*.jpg'):
    flow_u = Image.open(flow_u).convert('P')
    # print(torch.max(torch.from_numpy(np.array(flow_u))))
    flow_u = final_im_transform(flow_u)
    flow_u = flow_u - torch.mean(flow_u)
    print(flow_u.shape)
    flow = torch.cat([flow_u, flow_u], dim=0)
    print(flow.shape)
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
