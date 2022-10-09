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

def get_EPIC_parser():
    parser = ArgumentParser()
    parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return vars(args)

config = get_EPIC_parser()
print(Path(config['resume']))

# img = cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/myflow_frames/P01_01/37/u/frame_0000007029.jpg')

# img0 = np.array(cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/positive_frames/P01_01/37/frame_0000007034.jpg'), dtype =np.int32)
# img1 = np.array(cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/positive_frames/P01_01/37/frame_0000007035.jpg'), dtype =np.int32)
# img2 = np.array(cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/positive_frames/P01_01/37/frame_0000007036.jpg'), dtype =np.int32)
# plt.figure()
# ax = seaborn.heatmap(np.sum(np.abs(img0 - img1), axis=2), cmap='coolwarm', vmin=0, vmax=1, )
# ax.get_figure().savefig('temp1.jpg')
# plt.clf()

# plt.figure()
# ax = seaborn.heatmap(np.sum(np.abs(img2 - img1), axis=2), cmap='coolwarm', vmin=0, vmax=1, )
# ax.get_figure().savefig('temp2.jpg')
# plt.clf()
# print(np.max(np.abs(img0-img1)))
# print(np.max(np.abs(img2-img1)))
# print(img0[0,0])

# p = [[[0.5,0.2],[0.4,0.3]], [[0.5,0.8],[0.6,0.7]]]
# log_q = [[[0.2,0.4], [0.3,0.5]], [[0.6,0.3],[0.4,0.1]]]
# kl_result = torch.zeros_like(torch.tensor(p))
# kl = 0
# for dim0 in range(2):
#     for dim1 in range(2):
#         for dim2 in range(2):
#             temp_p = p[dim0][dim1][dim2]
#             temp_logq = log_q[dim0][dim1][dim2]
#             kl += temp_p * (np.log(temp_p) - temp_logq)
#             kl_result[dim0, dim1, dim2] = temp_p * (np.log(temp_p) - temp_logq)
# print(kl)
# print(kl_result)
# print(F.kl_div(torch.tensor(log_q), torch.tensor(p), reduction='none'))
# os.makedirs('/home/venom/projects/XMem/wandb', exist_ok=True)
# os.system(f'setx debug qnmlgcb')

# x = torch.tensor([[[0,1,2],
#                    [2,1,2]],
#                   [[0,1,2],
#                   [2,1,2]]], dtype=torch.long)
# print(F.one_hot(x, num_classes=3).permute(0,3,1,2))
# print(F.one_hot(x, num_classes=3).shape)
# print(x.shape)
# print(os.getcwd())
# os.chdir('./XMem_evaluation')
# print(os.getcwd())


# 从测试集的每一个video中随即选取一张mask，然后把每一个iter预测的这张mask都存下来，可视化结果
# def get_eval_pics(yaml_root, output_path, val_data_path, iterations):
#     with open(yaml_root, 'r') as f:
#         info = yaml.safe_load(f)
#     selected_pics = {} # {video_key: {gt_path:gt, rgb_path:rgb, pred_path: [pred1, pred2, ...]}}
#     for key, value in info.items():
#         partition = key.split('_')[0]
#         video_id = '_'.join(key.split('_')[:2])
#         anno_path = f'{val_data_path}/{partition}/anno_masks/{video_id}/{key}'
#         anno_pics = [pic.split('/')[-1] for pic in sorted(glob(f'{anno_path}/*.png'))[1:]] # [frame_0000xxx.png]
#         selected_pic = random.choice(anno_pics)
#         pred_masks = []
#         for it in iterations:
#             eval_it_path = f'{output_path}/eval_{it}'
#             pred_mask_path = f'{eval_it_path}/{partition}/{video_id}/{key}/{selected_pic}'
#             pred_masks.append(pred_mask_path)
#         rgb_path = [f'{val_data_path}/{partition}/rgb_frames/{video_id}/{key}/{selected_pic.replace("png", "jpg")}']*len(pred_masks)
#         gt_path = [f'{val_data_path}/{partition}/anno_masks/{video_id}/{key}/{selected_pic}']*len(pred_masks)
#         selected_pics.update({key: {'rgb_path': rgb_path, 'gt_path': gt_path, 'pred_path': pred_masks}})
#     return selected_pics
# wandb.init(project='0925_state_change_segm', group='debug', name='1004_debug_imgs')
# selected_pics = get_eval_pics(yaml_root='/home/venom/data/EPIC_val_split/EPIC100_state_positive_val.yaml', 
#                             output_path='/home/venom/.exp/0925_state_change_segm/1005_test_debug/debug_run',
#                             val_data_path='/home/venom/data/EPIC_val_split',
#                             iterations=[500,1000,1500])
# output_imgs = pair_pics_together(selected_pics)
# for img in output_imgs:
#     # print(sys.getsizeof(img))
#     # plt.imshow(img)
#     # plt.show()
#     img = wandb.Image(img)
#     wandb.log({"eval_imgs": img})

# wandb.finish()

# test_step = wandb.define_metric('test_step')
# wandb.define_metric(name='eval_acc', step_metric=test_step)
# for i in range(10):
#     wandb.log({'loss': i})
# for i in range(10):
#     wandb.log({'eval_loss': i})
    
# for i in range(10):
#     wandb.log({'test_step': i*10, 'eval_acc': i})
#     # wandb.log({})
#     # test_step += 1
# wandb.finish()
