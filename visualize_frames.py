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

mask_root_path = '/u/ryanxli/venom/0228_aot-benchmark_eval/saves/EPIC/EPIC_val_0815_openword_wo_tune_SwinB_DeAOTL_PRE_ckpt_unknown/all'
frame_root_path = '/u/ryanxli/venom/DeAOT/datasets/EPIC_val'
video_key = 'P22_01_164'
PART = video_key.split('_')[0]
VIDEO_ID = '_'.join(video_key.split('_')[:2])
save_path = '/u/ryanxli/venom/XMem/visuals/raw_DeAOT_img_visual'
for mask_path in glob.glob(f'{mask_root_path}/{PART}/{VIDEO_ID}/{video_key}/*.png'):
    frame_path = f'{frame_root_path}/{PART}/rgb_frames/{VIDEO_ID}/{video_key}/' + mask_path.split('/')[-1].replace('.png', '.jpg')
    frame = np.array(Image.open(frame_path))
    mask = np.array(Image.open(mask_path))
    visualization = overlay_davis(frame, mask)
    print(mask_path)
    os.makedirs(f"{save_path}/{PART}/{VIDEO_ID}/{video_key}", exist_ok=True)
    plt.imsave(f"{save_path}/{PART}/{VIDEO_ID}/{video_key}/{mask_path.split('/')[-1].replace('.png', '.jpg')}", visualization)
    