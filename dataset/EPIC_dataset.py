import os
from os import path, replace
import math
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import sys
sys.path.append('/cluster/home2/yjw/venom/XMem')
from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed
import yaml
import matplotlib.pyplot as plt
from glob import glob
import torch.nn as nn

class EPICDataset(Dataset):
    """
    Works for EPIC training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, data_root, yaml_root, max_jump, num_frames=3, max_num_obj=3, finetune=False):
        print('We are using EPIC Dataset !!!!!')
        self.data_root = data_root
        self.max_jump = max_jump
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj
        with open(os.path.join(yaml_root), 'r') as f:
            self.data_info = yaml.safe_load(f)
        self.vids = [] 
        for key in list(self.data_info.keys()):
            PART = key.split('_')[0]
            VIDEO_ID = '_'.join(key.split('_')[:2])
            vid_gt_path = os.path.join(self.data_root, PART, 'anno_masks', VIDEO_ID, key)
            # print(vid_gt_path)
            # print(glob(vid_gt_path))
            
            if len(glob(f"{vid_gt_path}/*.png")) >= 2:
                self.vids.append(key)
        
        assert num_frames >= 3
        
    def __getitem__(self, idx):
        video_value = self.data_info[self.vids[idx]] # video value
        
        info = {}
        info['name'] = self.vids[idx]

        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], self.vids[idx])
        # first last frame
        vid_gt_path = path.join(self.data_root, video_value['participant_id'], 'anno_masks', video_value['video_id'], self.vids[idx])
        vid_flow_path = path.join(self.data_root, video_value['participant_id'], 'flow_frames', video_value['video_id'], self.vids[idx])
        frames = list(range(video_value['start_frame'], video_value['stop_frame']))

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.num_frames
            # video有多少帧图片
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            # 随机选取一帧，然后选取这一帧图片前后各自max jump以内的图片
            # 这样保证了得到的采样帧两帧之间不会相差超过max_jump
            # 每选取一帧，就把该帧前后的max_jump帧都append进去
            frames_idx = [0, len(frames)-1, np.random.randint(1,length-1)] # first, last, random
            acceptable_set = set(range(max(1, frames_idx[-1]-this_max_jump), min(length-1, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(1, frames_idx[-1]-this_max_jump), min(length-1, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            
            # frames_idx就是sample出来的帧的索引
            images = []
            masks = []
            flows = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.jpg'
                png_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.png'
                if len(video_value['video_id'].split('_')[-1]) == 2:
                    flow_idx = int(np.ceil((float(frames[f_idx]) - 3) / 2))
                    flow_name = 'frame_' + str(int(np.ceil((float(frames[f_idx]) - 3) / 2))).zfill(10)+ '.jpg'
                    while True:
                        if os.path.exists(path.join(vid_flow_path, 'u', flow_name)):
                            break
                        else:
                            for i in range(100):
                                flow_name = 'frame_' + str(flow_idx - i).zfill(10)+ '.jpg'
                                if os.path.exists(path.join(vid_flow_path, 'u', flow_name)):
                                    left_flow_idx = flow_idx - i
                                    break
                                else:
                                    left_flow_idx = flow_idx - i
                            for i in range(100):
                                flow_name = 'frame_' + str(flow_idx + i).zfill(10)+ '.jpg'
                                if os.path.exists(path.join(vid_flow_path, 'u', flow_name)):
                                    right_flow_idx = flow_idx + i
                                    break
                                else:
                                    right_flow_idx = flow_idx + i
                            # print(np.abs(right_flow_idx - flow_idx), np.abs(flow_idx - left_flow_idx))
                            if np.minimum(np.abs(right_flow_idx - flow_idx), np.abs(left_flow_idx - flow_idx)) > 20:
                                print('Warning: flow frame too large')
                            if np.abs(right_flow_idx - flow_idx) > np.abs(left_flow_idx - flow_idx):
                                flow_idx = left_flow_idx
                            else:
                                flow_idx = right_flow_idx
                            
                            flow_name = 'frame_' + str(flow_idx).zfill(10)+ '.jpg'
                            break 
                        # raise ValueError(f"flow file not exist:{path.join(vid_flow_path, 'u', flow_name)}")
                else:
                    flow_idx = f_idx
                    flow_name = 'frame_' + str(frames[flow_idx]).zfill(10)+ '.jpg'
                info['frames'].append(jpg_name)
                
                images.append(np.array(Image.open(path.join(vid_im_path, jpg_name)))) #####
                
                # aggregate flow
                agg_u_frames = []
                agg_v_frames = []
                u_path = path.join(vid_flow_path, 'u')
                v_path = path.join(vid_flow_path, 'v')
                all_u_jpgs = sorted(glob(f'{u_path}/*.jpg'))
                all_v_jpgs = sorted(glob(f'{v_path}/*.jpg'))
                assert len(all_u_jpgs) > 5 and len(all_v_jpgs) > 5
                u_idx = all_u_jpgs.index(path.join(vid_flow_path, 'u', flow_name))
                v_idx = all_v_jpgs.index(path.join(vid_flow_path, 'v', flow_name))
                if u_idx == 0 or u_idx == 1:
                    agg_u_frames = all_u_jpgs[:5]
                elif u_idx == len(all_u_jpgs) - 1 or u_idx == len(all_u_jpgs) - 2:
                    agg_u_frames = all_u_jpgs[-5:]
                else:
                    agg_u_frames = all_u_jpgs[u_idx-2:u_idx+3]
                
                if v_idx == 0 or v_idx == 1:
                    agg_v_frames = all_v_jpgs[:5]
                elif v_idx == len(all_v_jpgs) - 1 or v_idx == len(all_v_jpgs) - 2:
                    agg_v_frames = all_v_jpgs[-5:]
                else:
                    agg_v_frames = all_v_jpgs[v_idx-2:v_idx+3]
                
                agg_flow = []
                for u_frame, v_frame in zip(agg_u_frames, agg_v_frames):
                    u = np.array(Image.open(u_frame))
                    v = np.array(Image.open(v_frame))
                    flow = np.stack((u,v), axis=2)
                    agg_flow.append(flow)
                flows.append(np.stack(agg_flow, 0))
                
                if f_idx == frames_idx[0] or f_idx == frames_idx[-1]:
                    this_gt = np.array(Image.open(path.join(vid_gt_path, png_name)).convert('1'))
                    
                    # this_gt = np.array(Image.fromarray(this_gt))
                    # print(f'ddfdf:{np.unique(this_gt)}')
                    # print(f'dataset:{np.unique(this_gt)}')
                    masks.append(this_gt)

            labels = np.unique(masks[0])
            # Remove background
            # mask的存储形式应该是，每一个pixel属于哪一类，0,1,2...，visualize的时候，每一个类给一个color就行了
            # labels里面是所有的类别0,1,2,3...，包括背景
            labels = labels[labels!=0]
            
            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break
        
        images = np.stack(images, 0)
        flows = np.stack(flows, 0)
        masks = np.stack(masks, 0)
        
        # 如果object数量太多就随机选
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)
        # 至少一个target object
        info['num_objects'] = max(1, len(target_objects))
        
        # 1 if object exist, 0 otherwise
        # list:len = max_num_obj, 前num_objects个是1，后面是0
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        # data = {
        #     'rgb': images, # [num_frames, 3, H, W]
        #     'forward_flow': forward_flows, # [num_frames, 10, H, W]
        #     'backward_flow': backward_flows, # [num_frames, 10, H, W]
        #     'first_last_frame_gt': first_last_frame_gt, # [2, max_num_obj, H, W] one hot
        #     'cls_gt': cls_gt, # [2, 1, H, W]
        #     'selector': selector, # [max_num_obj] 前num_objects个是1，后面是0
        #     'text':video_value['narration'],
        #     'info': info,
        # }
        target_objects = torch.tensor(target_objects, dtype=torch.bool)
        
        data = {
            'rgb': images, # [num_frames, H, W, 3]
            'flows': flows, # [num_frames, 5, H, W, 2]
            'masks': masks, # [2, H, W]
            'target_objects': target_objects, # [num_objects]
            'selector': selector, # [max_num_obj] 前num_objects个是1，后面是0
            'text': video_value['narration'],
            'info': info,
            'action_label': video_value['verb_class'],
        }

        return data

    def __len__(self):
        return len(self.vids)

if __name__ == '__main__':
    dataset = EPICDataset(data_root='../data', yaml_root='../data/EPIC55_cut_subset_200.yaml', max_jump=20, num_frames=3, max_num_obj=3, finetune=False)
    images = dataset[2]
    print(f"name={images['info']['name']}")
    
    for obj in range(images['first_last_frame_gt'].shape[1]):
        plt.imsave(f"../visuals/gt_{obj}.jpg", images['first_last_frame_gt'][0,obj],cmap='gray')