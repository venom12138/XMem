import os
from os import path, replace

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
from torch.utils.data import DataLoader

class EPICtestDataset(Dataset):
    """
    Works for EPIC training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, data_root, yaml_root, max_num_obj=3, finetune=False):
        self.data_root = data_root
        self.max_num_obj = max_num_obj
        with open(os.path.join(yaml_root), 'r') as f:
            self.data_info = yaml.safe_load(f)
        self.vids = list(self.data_info.keys())
        # 将没有标注的都去掉
        for k in list(self.data_info.keys()):
            video_value = self.data_info[k]
            vid_gt_path = path.join(self.data_root, video_value['participant_id'], 'anno_masks', video_value['video_id'], k)
            frame_name = video_value['start_frame']
            jpg_name = 'frame_' + str(frame_name).zfill(10)+ '.jpg'
            png_name = 'frame_' + str(frame_name).zfill(10)+ '.png'
            if not os.path.isfile(os.path.join(vid_gt_path, jpg_name)) and not os.path.isfile(os.path.join(vid_gt_path, png_name)):
                self.vids.remove(k)
        # Final transform without randomness
        self.im_transform = transforms.Compose([
            transforms.Resize((384,384), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((384,384), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        # 获取第一帧图片的大小
        
        video_value = self.data_info[self.vids[0]] # video value
        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], self.vids[0])
        frames = list(range(video_value['start_frame'], video_value['stop_frame']))
        jpg_name = 'frame_' + str(frames[1]).zfill(10)+ '.jpg'
        this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
        self.img_size = [this_im.size[1], this_im.size[0]] # 456,256


    def __getitem__(self, idx):
        video_value = self.data_info[self.vids[idx]] # video value
        
        info = {}
        info['name'] = self.vids[idx]
        info['frames'] = []
        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], self.vids[idx])
        # first last frame
        vid_gt_path = path.join(self.data_root, video_value['participant_id'], 'anno_masks', video_value['video_id'], self.vids[idx])
        vid_flow_path = path.join(self.data_root, video_value['participant_id'], 'flow_frames', video_value['video_id'], self.vids[idx])
        frames = list(range(video_value['start_frame'], video_value['stop_frame']))
        
        sequence_seed = np.random.randint(2147483647)
        images = []
        masks = []
        masks_count = [] # 标记是否当前帧是否有标注的annotation
        flows = []
        target_objects = []
        
        for f_idx in range(len(frames)):
            jpg_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.jpg'
            png_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.png'
            if not os.path.isfile(path.join(vid_gt_path, png_name)):
                if f_idx % 2 == 0:
                    continue
                    # pass
            if len(video_value['video_id'].split('_')[-1]) == 2:
                flow_name = 'frame_' + str(int(np.ceil((float(frames[f_idx]) - 3) / 2))).zfill(10)+ '.jpg'
            else:
                flow_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.jpg'
            info['frames'].append(jpg_name)

            # 需要保证image、gt和flow做同样的变换，要不然mask就对不上了
            reseed(sequence_seed)
            this_im = Image.open(path.join(vid_im_path, jpg_name))# .convert('RGB')
            this_im = self.im_transform(this_im)

            reseed(sequence_seed)
            this_flowu = Image.open(path.join(vid_flow_path, 'u', flow_name)).convert('P').resize((384,384))

            reseed(sequence_seed)
            this_flowv = Image.open(path.join(vid_flow_path, 'v', flow_name)).convert('P').resize((384,384))

            if os.path.isfile(path.join(vid_gt_path, png_name)):
                masks_count.append(1)
                if f_idx == 0:
                    reseed(sequence_seed)
                    this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('1')
                    this_gt = self.gt_transform(this_gt)                
                    this_gt = this_gt.squeeze()
                    masks.append(this_gt)
                    labels = np.unique(this_gt)
            else:
                masks_count.append(0)

            this_flow = torch.stack([torch.from_numpy(np.array(this_flowu)), torch.from_numpy(np.array(this_flowv))], dim=0)
            
            images.append(this_im)
            flows.append(this_flow)
        
        images = torch.stack(images, 0)
        flows = torch.stack(flows, 0).float()
        masks_count = torch.tensor(masks_count, dtype=torch.int)
        # Remove background
        # mask的存储形式应该是，每一个pixel属于哪一类，0,1,2...，visualize的时候，每一个类给一个color就行了
        # labels里面是所有的类别0,1,2,3...，包括背景
        labels = labels[labels!=0]
        
        if len(labels) == 0:
            target_objects = []
        else:
            target_objects = labels.tolist()
            
        
        # 如果object数量太多就随机选
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)
        # 至少一个target object
        info['num_objects'] = max(1, len(target_objects))
        # 这相当于是把list，stack成np array
        # masks是一个[1, H, W]的np array
        # masks 保持为一个list，因为里面存在none
        masks = np.stack(masks, 0)
        assert masks.shape[0] == 1

        # Generate one-hot ground-truth
        cls_gt = np.zeros((1, 384, 384), dtype=np.int32) # 只有1帧有mask
        first_frame_gt = np.zeros((1, len(target_objects), 384, 384), dtype=np.int32)
        
        # target_objects是一个list，长度是objects的数量
        for i, l in enumerate(target_objects):
            # masks是一个[1, H, W]的np array
            # this_mask一个[1, H, W]的np array, 其中每个像素值是true or false
            this_mask = (masks==l)
            # cls_gt是一个[2, H, W]的np array，将cls_gt和this_mask对应的位置赋上值
            try:
                cls_gt[this_mask] = i+1
            except:
                print(cls_gt.shape)
                print(this_mask.shape)
                print(masks.shape)
                print(l)
                print(i)
                print(self.vids[idx])
                raise Exception('error')
            # first_frame_gt是一个one hot向量，[1, num_objects, H, W]
            # 将所有的num_frame里面的第一个，也就是第一个frame，里的第i个object赋给first_frame_gt，也就是一个0,1的array
            first_frame_gt[:,i] = this_mask
        # expand完变成(1, 1, H, W)
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        # list:len = max_num_obj, 前num_objects个是1，后面是0
        selector = [1 if i < info['num_objects'] else 0 for i in range(len(target_objects))]
        selector = torch.FloatTensor(selector)
        
        data = {
            'rgb': images, # [num_frames, 3, H, W]
            'flow': flows, # [num_frames, 2, H, W]
            'first_frame_gt': first_frame_gt, # [1, target_objects, H, W] one hot
            'cls_gt': cls_gt, # [1, 1, H, W]
            'selector': selector, # [target_objects] 前num_objects个是1，后面是0
            'info': info,
            'whether_save_mask': masks_count,
        }

        return data

    def __len__(self):
        return len(self.vids)

if __name__ == '__main__':
    dataset = EPICtestDataset(data_root='../data', yaml_root='../data/EPIC55_cut_subset_200.yaml', max_num_obj=3, finetune=False)
    val_loader = DataLoader(dataset, 1,  shuffle=False, num_workers=4)
    for i, data in enumerate(val_loader):
        print(data['info']['name'][0])
        print(data['rgb'][0].shape)
        print(data['first_frame_gt'][0][0].shape)
        print(data['whether_save_mask'][0][1])
        # print(data[])
        dd
    
    # images = dataset[2]
    # print(f"name={images['first_frame_gt'].shape}")
    
    # for obj in range(images['first_frame_gt'].shape[1]):
    #     plt.imsave(f"../visuals/gt_{obj}.jpg", images['first_frame_gt'][0,obj],cmap='gray')