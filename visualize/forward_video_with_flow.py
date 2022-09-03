import torch
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append('..')
if torch.cuda.is_available():
  print('Using GPU')
  device = 'cuda'
else:
  print('CUDA not available. Please connect to a GPU instance if possible.')
  device = 'cpu'

import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar
import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.set_grad_enabled(False)

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

ROOT_PATH = '../data'
vid = 'P01_01_37'
partition_id = vid.split('_')[0]
video_id = partition_id + '_' + vid.split('_')[1]

ckpt_path = '../saves/Aug31_11.59.01_test_0831_epic_50000.pth'
network = XMem(config, ckpt_path).eval().to(device)

mask_save_path = f'../visuals/{partition_id}/flow_mask/{video_id}/{vid}'
draw_save_path = f'../visuals/{partition_id}/flow_draw/{video_id}/{vid}'

video_path = f'{ROOT_PATH}/{partition_id}/rgb_frames/{video_id}/{vid}'
u_flow_path = f'{ROOT_PATH}/{partition_id}/flow_frames/{video_id}/{vid}/u'
v_flow_path = f'{ROOT_PATH}/{partition_id}/flow_frames/{video_id}/{vid}/v'

# use first mask
mask_name = sorted(glob.glob(f'{ROOT_PATH}/{partition_id}/anno_masks/{video_id}/{vid}/*.jpg'))[0]

if not os.path.isdir(mask_save_path):
    os.makedirs(mask_save_path)
if not os.path.isdir(draw_save_path):
    os.makedirs(draw_save_path)

mask = np.array(Image.open(mask_name).convert('1'),dtype=np.int32)
print(np.unique(mask))
print(mask.shape)
num_objects = len(np.unique(np.round(mask))) - 1

"""# Propagte frame-by-frame"""

torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
# cap = cv2.VideoCapture(video_name)

# You can change these two numbers
frames_to_propagate = 1000
visualize_every = 20

current_frame_index = 0

with torch.cuda.amp.autocast(enabled=True):
    for frame_path in sorted(glob.glob(f'{video_path}/*.jpg')):
        jpg_name = frame_path.split('/')[-1]
        if len(vid.split('_')[1]) == 2:
            flow_id = int(jpg_name.split('.')[0].split('_')[1])
            flow_name = 'frame_' + str(int(np.ceil((float(flow_id) - 3) / 2))).zfill(10)+ '.jpg'
            u_flow_name = f'{u_flow_path}/{flow_name}'
            v_flow_name = f'{v_flow_path}/{flow_name}'
            u_flow = np.array(Image.open(u_flow_name).convert('P'))
            v_flow = np.array(Image.open(v_flow_name).convert('P'))
            flow = np.stack([u_flow, v_flow], axis=0)
        else:
            raise NotImplementedError
        # load frame-by-frame
        frame = np.array(Image.open(frame_path))
        # plt.imsave(f"{draw_save_path}/{uid}/{frame_path.split('/')[-1]}", frame)
        print(frame_path)
        if frame is None or current_frame_index > frames_to_propagate:
            break

        # convert numpy array to pytorch tensor format
        frame_torch, _ = image_to_torch(frame, device=device)

        flow = flow.transpose(0, 1, 2)
        # print(f'flow_shape:{flow.shape}')
        flow_torch = torch.from_numpy(flow).float().to(device)# /255
        
        if current_frame_index == 0:
            # initialize with the mask
            mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
            
            # the background mask is not fed into the model
            prediction = processor.step(frame_torch, flow_torch, mask_torch[1:])
        else:
            # propagate only
            prediction = processor.step(frame_torch, flow_torch)

        # argmax, convert to numpy
        # 0,1
        prediction = torch_prob_to_numpy_mask(prediction)
        
        # plt.imsave(f"{mask_save_path}/{uid}/{frame_path.split('/')[-1]}", prediction*255)
        
        # if current_frame_index % visualize_every == 0:
        visualization = overlay_davis(frame, prediction)
            # print(prediction.shape)
            # print(visualization.shape)
        plt.imsave(f"{draw_save_path}/{frame_path.split('/')[-1]}", visualization)

        current_frame_index += 1
import imageio.v2 as imageio
images = []
for frame_path in sorted(glob.glob(f'{draw_save_path}/*.jpg')):
    im = imageio.imread(frame_path)
    images.append(im)
if not os.path.exists(f'../visuals/{partition_id}/flow_gif/{video_id}/'):
    os.makedirs(f'../visuals/{partition_id}/flow_gif/{video_id}/')
    
imageio.mimsave(f'../visuals/{partition_id}/flow_gif/{video_id}/{vid}.gif', images, 'GIF', duration=0.05)
# f'../visuals/{partition_id}/flow_draw/{video_id}/{vid}'