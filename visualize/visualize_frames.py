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

for mask_path in glob.glob('/home/venom/projects/fbrs_normal/output/P01/rgb_frames/P01_01/P01_01_37/*.png'):
    frame_path = '/home/venom/projects/XMem/EPIC_train/P01/rgb_frames/P01_01/P01_01_37/' + mask_path.split('/')[-1].replace('.png', '.jpg')
    frame = np.array(Image.open(frame_path))
    mask = np.array(Image.open(mask_path))
    visualization = overlay_davis(frame, mask)
    plt.imsave(f"./{mask_path.split('/')[-1].replace('.png', '.jpg')}", visualization)