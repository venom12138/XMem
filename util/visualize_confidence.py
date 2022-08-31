import torch
import glob
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
import matplotlib.pyplot as plt
import seaborn

# seaborn.set(rc = {'figure.figsize':(456,256)})
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

network = XMem(config, './saves/XMem.pth').eval().to(device)

conf_save_path = './data/P04/conf_masks/P04_01'
# draw_save_path = './data/P04/conf_draws/P04_01'
video_path = '/cluster/home2/yjw/venom/XMem/data/P04/positive_frames/P04_01/7967'
# use first mask
mask_name = '/cluster/home2/yjw/venom/EPIC-data/data/P04/first_last_masks/P04_01/7967/frame_0000012756.jpg'
uid = video_path.split('/')[-1]

if not os.path.isdir(f"{conf_save_path}/{uid}"):
    os.makedirs(f"{conf_save_path}/{uid}")

# if not os.path.isdir(f"{draw_save_path}/{uid}"):
#     os.makedirs(f"{draw_save_path}/{uid}")

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
        # load frame-by-frame
        frame = np.array(Image.open(frame_path))
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
            prediction = processor.step(frame_torch, mask_torch[1:])
        else:
            # propagate only
            prediction = processor.step(frame_torch)

        # argmax, convert to numpy
        # 0,1
        # 进去之前的prediction是一个[图层数，高，宽]的tensor，通过argmax选出来每一个pixel的归属，属于哪一个图层
        prediction = torch.abs(prediction[0] - prediction[1]).cpu().numpy()
        # print(prediction.shape)
        # ddddd
        # prediction = torch_prob_to_numpy_mask(prediction)
        # print(prediction.shape)
        plt.figure()
        ax = seaborn.heatmap(prediction, cmap='coolwarm', vmin=0, vmax=1, )
        ax.get_figure().savefig(f"{conf_save_path}/{uid}/{frame_path.split('/')[-1]}")
        plt.clf()
        # plt.imsave(f"{conf_save_path}/{uid}/{frame_path.split('/')[-1]}", prediction*255)
        
        # if current_frame_index % visualize_every == 0:
        # visualization = overlay_davis(frame, prediction)
            # print(prediction.shape)
            # print(visualization.shape)
        # plt.imsave(f"{draw_save_path}/{uid}/{frame_path.split('/')[-1]}", visualization)

        current_frame_index += 1

# import imageio.v2 as imageio
# images = []
# for frame_path in sorted(glob.glob(f'{draw_save_path}/{uid}/*.jpg')):
#     im = imageio.imread(frame_path)
#     images.append(im)
# imageio.mimsave(f"/cluster/home2/yjw/venom/EPIC-data/data/P04/conf_gif/{uid}.gif", images, 'GIF', duration=0.05)