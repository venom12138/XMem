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
# palette = [0,1,2,3,4,5,6]
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    # h,w = new_mask.size
    palette = [0,0,0,255,255,255,128,0,0,0,128,0,0,0,128,255,0,0,255,255,0]
    others = list(np.random.randint(0,255,size=256*3-len(palette)))
    palette.extend(others)
    new_mask.putpalette(palette)

    return new_mask
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

network = XMem(config, '../saves/XMem.pth').eval().to(device)

uid = 'P01_01_37'
part = uid.split('_')[0]
video_id = '_'.join(uid.split('_')[:2])
mask_save_path = f'../visuals/mid_masks/{part}/{video_id}'
draw_save_path = f'../visuals/mid_draws/{part}/{video_id}'
video_path = f'/home/venom/projects/XMem/data/{part}/rgb_frames/{video_id}/{uid}'
# use first mask
mask_name = f'/home/venom/projects/XMem/data/{part}/anno_masks/{video_id}/{uid}/frame_0000007067.png'
uid = video_path.split('/')[-1]

if not os.path.isdir(f"{mask_save_path}/{uid}"):
    os.makedirs(f"{mask_save_path}/{uid}")
if not os.path.isdir(f"{draw_save_path}/{uid}"):
    os.makedirs(f"{draw_save_path}/{uid}")

mask = np.array(Image.open(mask_name).convert('1').resize((384,384)),dtype=np.int32)
print(np.unique(mask))
print(mask.shape)
num_objects = len(np.unique(np.round(mask))) - 1

"""# Propagte frame-by-frame"""

torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
# cap = cv2.VideoCapture(video_name)

# You can change these two numbers
frames_to_propagate = 800
visualize_every = 20

frame_list = sorted(glob.glob(f'{video_path}/*.jpg'))
# print(frame_list)
mid_pos = frame_list.index(mask_name.replace('anno_masks','rgb_frames').replace('.png','.jpg'))

frames_before = frame_list[:mid_pos+1]
frames_after = frame_list[mid_pos:]

with torch.cuda.amp.autocast(enabled=True):
    current_frame_index = 0
    for frame_path in list(reversed(frames_before)):
        # load frame-by-frame
        frame = np.array(Image.open(frame_path).resize((384,384)))
        frame_raw = np.array(Image.open(frame_path))
        # plt.imsave(f"{draw_save_path}/{uid}/{frame_path.split('/')[-1]}", frame)
        print(frame_path)
        if frame is None or current_frame_index > frames_to_propagate:
            break

        # convert numpy array to pytorch tensor format
        frame_torch, _ = image_to_torch(frame, device=device)
        if current_frame_index == 0:
            # initialize with the mask
            mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
            
            # the background mask is not fed into the model
            prediction = processor.step(frame_torch, mask = mask_torch[1:])
        else:
            # propagate only
            prediction = processor.step(frame_torch)
        prediction = F.interpolate(prediction.unsqueeze(1), (256,456), mode='bilinear', align_corners=False)[:,0]
        # argmax, convert to numpy
        # 0,1
        prediction = torch_prob_to_numpy_mask(prediction)
        mask_img = colorize_mask(prediction)
        mask_img.save(f"{mask_save_path}/{uid}/{frame_path.split('/')[-1].replace('jpg', 'png')}")        
        # plt.imsave(f"{mask_save_path}/{uid}/{frame_path.split('/')[-1]}", prediction*255)
        
        # if current_frame_index % visualize_every == 0:
        visualization = overlay_davis(frame_raw, prediction)
            # print(prediction.shape)
            # print(visualization.shape)
        plt.imsave(f"{draw_save_path}/{uid}/{frame_path.split('/')[-1]}", visualization)

        current_frame_index += 1

    current_frame_index = 0
    for frame_path in frames_after:
        # load frame-by-frame
        frame = np.array(Image.open(frame_path).resize((384,384)))
        frame_raw = np.array(Image.open(frame_path))
        # plt.imsave(f"{draw_save_path}/{uid}/{frame_path.split('/')[-1]}", frame)
        print(frame_path)
        
        if frame is None or current_frame_index > frames_to_propagate:
            break

        # convert numpy array to pytorch tensor format
        frame_torch, _ = image_to_torch(frame, device=device)
        if current_frame_index == 0:
            # initialize with the mask
            mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
            
            # the background mask is not fed into the model
            prediction = processor.step(frame_torch, mask = mask_torch[1:])
        else:
            # propagate only
            prediction = processor.step(frame_torch)
        prediction = F.interpolate(prediction.unsqueeze(1), (256,456), mode='bilinear', align_corners=False)[:,0]
        # argmax, convert to numpy
        # 0,1
        prediction = torch_prob_to_numpy_mask(prediction)
        mask_img = colorize_mask(prediction)
        mask_img.save(f"{mask_save_path}/{uid}/{frame_path.split('/')[-1].replace('jpg', 'png')}")     
        # plt.imsave(f"{mask_save_path}/{uid}/{frame_path.split('/')[-1]}", prediction*255)
        
        # if current_frame_index % visualize_every == 0:
        visualization = overlay_davis(frame_raw, prediction)
            # print(prediction.shape)
            # print(visualization.shape)
        plt.imsave(f"{draw_save_path}/{uid}/{frame_path.split('/')[-1]}", visualization)

        current_frame_index += 1
        
# import imageio.v2 as imageio
# images = []
# for frame_path in sorted(glob.glob(f'{draw_save_path}/{uid}/*.jpg')):
#     im = imageio.imread(frame_path)
#     images.append(im)
# imageio.mimsave(f"/cluster/home2/yjw/venom/EPIC-data/data/P04/mid_gif/{uid}.gif", images, 'GIF', duration=0.05)