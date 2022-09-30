import cv2
import numpy as np
import glob
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

wandb.init(project='0925_state_change_segm', group='debug', name='0930_debug_step')
test_step = wandb.define_metric('test_step')
wandb.define_metric(name='eval_acc', step_metric=test_step)
for i in range(10):
    wandb.log({'loss': i})
for i in range(10):
    wandb.log({'eval_loss': i})
    
for i in range(10):
    wandb.log({'test_step': i*10, 'eval_acc': i})
    # wandb.log({})
    # test_step += 1
wandb.finish()