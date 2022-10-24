import sys
sys.path.append('..')
from model.mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
from tqdm import tqdm
from PIL import Image
import numpy as np 
from skimage.io import imsave
import pdb
from copy import deepcopy
import torch
from torchvision import transforms
from model.range_transform import im_normalization, im_mean
from torchvision.transforms import InterpolationMode
import random

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

class TestDataPreprocess():
    def __init__(self, config_file='saves/seg_twohands_ccda/seg_twohands_ccda.py', \
                checkpoint_file='saves/seg_twohands_ccda/best_mIoU_iter_56000.pth', \
                max_num_obj=3, remove_hand=True):
        self.remove_hand = remove_hand
        self.max_num_obj = max_num_obj
        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(config_file, checkpoint_file, device=f'cuda')

        # Final transform without randomness
        self.im_transform = transforms.Compose([
            transforms.Resize((384,384), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((384,384), interpolation=InterpolationMode.NEAREST),
        ])
    
    def preprocess(self, data):
        # images:[1, H, W, 3]
        images = data['rgb'] 
        # flows:[1, 5, H, W, 2]: 2 for u and v, 5 for 5 neighbor flow
        flows = data['flows']
        # masks:[1, H, W]: 2 for first
        if 'masks' in list(data.keys()):
            masks = data['masks']
        else:
            masks = None
        target_objects = data['target_objects']
        B = images.shape[0]
        H = 384
        W = 384
        
        new_rgbs = torch.zeros((B, 3, H, W), dtype=torch.float32)
        new_forward_flows = torch.zeros((B, 10, H, W), dtype=torch.float32)
        
        # get hand regions
        if self.remove_hand:
            hand_regions = self.get_hand_region(images)

        for bi in range(B):
            # important:因为现在是convert 1， 所以现在的target objects只有False和True，后期要是改成很多objects，就要convert P
            # target_objects 是0，1，2这样子
            sequence_seed = np.random.randint(2147483647)
            pairwise_seed = np.random.randint(2147483647)
            
            img = Image.fromarray(images[bi].cpu().numpy(), mode='RGB')
            agg_flow = flows[bi] # [5, H, W, 2]
            this_im = self.img_preprocess(img, sequence_seed, pairwise_seed)
            # forward_flow: [10, H, W]
            if self.remove_hand:
                this_forward_flow = self.flow_preprocess(agg_flow, sequence_seed, pairwise_seed, hand_mask=hand_regions[bi])
            else:
                this_forward_flow = self.flow_preprocess(agg_flow, sequence_seed, pairwise_seed, hand_mask=None)
            
            new_rgbs[bi]  = this_im
            new_forward_flows[bi] = this_forward_flow
            if masks is not None:
                video_mask = []
                this_gt = masks[bi]
                this_gt = self.mask_preprocess(this_gt, sequence_seed, pairwise_seed)
                video_mask.append(this_gt)
                video_mask = np.stack(video_mask)
                
                first_frame_gt = np.zeros((1, len(target_objects), 384, 384), dtype=np.int32)

                # target_objects是一个list，长度是objects的数量
                for i, l in enumerate(target_objects):
                    # masks是一个[1, H, W]的np array
                    # this_mask一个[1, H, W]的np array, 其中每个像素值是true or false
                    this_mask = (video_mask==l)
                    
                    # first_frame_gt是一个one hot向量，[1, num_objects, H, W]
                    # 将所有的num_frame里面的第一个，也就是第一个frame，里的第i个object赋给first_frame_gt，也就是一个0,1的array
                    first_frame_gt[:,i] = this_mask
                
        # print(f"dfd: {data['target_objects']}")
        # print(f"images: {new_rgbs.shape}")
        # print(f"forward flow: {new_forward_flows.shape}")
        # print(f"backward flow: {new_backward_flows.shape}")
        # print(f"new_first_frame_gt: {new_first_frame_gt.shape}")
        # print(f"mask: {this_gt.shape}")
        
        if masks is not None:
            new_data = {
                'rgb': new_rgbs.cuda(), # [B, 3, H, W]
                'forward_flow': new_forward_flows.cuda(), # [B, 10, H, W]
                # 'backward_flow': new_backward_flows.cuda(), # [B, 10, H, W]
                'first_frame_gt': torch.tensor(first_frame_gt).cuda().to(torch.long), # [1, target_objects, H, W] one hot
                'selector': data['selector'], 
                'info': data['info'],
                'whether_save_mask': data['whether_save_mask'],
            }
        else:
            new_data = {
                'rgb': new_rgbs.cuda(), # [B, 3, H, W]
                'forward_flow': new_forward_flows.cuda(), # [B, 10, H, W]
                # 'backward_flow': new_backward_flows.cuda(), # [B, 10, H, W]
                'selector': data['selector'], 
                'info': data['info'],
                'whether_save_mask': data['whether_save_mask'],
            }
            
        return new_data
        
    def mask_preprocess(self, mask, sequence_seed, pairwise_seed):
        this_gt = Image.fromarray(mask)
        this_gt = self.gt_transform(this_gt)
        
        return np.array(this_gt)
    
    # input: imgs: [B, num_frames, H, W, 3]
    # return: [B, num_frames, H, W]
    def get_hand_region(self, imgs):
        if len(imgs.shape) == 5:
            B = imgs.shape[0]
            num_frames = imgs.shape[1]
            imgs = imgs.reshape(B*num_frames, *imgs.shape[2:])
            need_reshape = True
        else:
            B = imgs.shape[0]
            need_reshape = False
        imgs = np.array(imgs.cpu())
        hand_regions = np.array(inference_segmentor(self.model, imgs)).astype(np.uint8)
        if need_reshape:
            hand_regions = hand_regions.reshape(B, num_frames, *hand_regions.shape[1:])
        
        return hand_regions

    def flow_preprocess(self, agg_flow, sequence_seed, pairwise_seed, hand_mask=None):
        forward_flow = []
        # backward_flow = []
        for ni in range(5):
            flowu = agg_flow[ni, :, :, 0]
            flowv = agg_flow[ni, :, :, 1]
            
            flowu = Image.fromarray(flowu, mode='P')
            flowv = Image.fromarray(flowv, mode='P')
            # backward_u = Image.fromarray(255 - np.array(flowu), mode='P')
            # backward_v = Image.fromarray(255 - np.array(flowv), mode='P')
            if self.remove_hand:
                flowu, flowv = self.remove_hand_region(np.array(hand_mask), np.array(flowu), np.array(flowv))
                # backward_u, backward_v = self.remove_hand_region(np.array(hand_mask), np.array(backward_u), np.array(backward_v))
            
            flowu = self.gt_transform(flowu)
            flowv = self.gt_transform(flowv)
            # backward_u = self.gt_transform(backward_u)
            # backward_v = self.gt_transform(backward_v)
            
            # print(f'somethingaboutflow:{torch.max(torch.tensor(np.array(flowu)))}')
            # 将0-255的像素值映射到0到1之间并中心化
            flowu = transforms.ToTensor()(flowu)
            flowv = transforms.ToTensor()(flowv)
            flowu = flowu - torch.mean(flowu)
            flowv = flowv - torch.mean(flowv)
            
            # 将0-255的像素值映射到0到1之间并中心化
            # backward_u = transforms.ToTensor()(backward_u)
            # backward_v = transforms.ToTensor()(backward_v)
            # backward_u = backward_u - torch.mean(backward_u)
            # backward_v = backward_v - torch.mean(backward_v)
            
            forward_flow.append(flowu.squeeze())
            forward_flow.append(flowv.squeeze())
            # backward_flow.append(backward_u.squeeze())
            # backward_flow.append(backward_v.squeeze())
            # print(f'somethingaboutflow:{torch.max(forward_flow[0])}')
            
        # [10, H, W]
        return torch.stack(forward_flow) # torch.stack(backward_flow)
    
    def img_preprocess(self, img, sequence_seed, pairwise_seed):
        img = self.im_transform(img)
        return img
    
    # img: nparray [HWC]
    # flow: nparray [HWC]
    # 将手的区域的flow置为0
    def remove_hand_region(self, hand_mask, flowu, flowv):
        # print(f'dddimg:{img.shape}')
        flowu = torch.tensor(flowu).to(torch.float64)
        flowv = torch.tensor(flowv).to(torch.float64)
        assert len(hand_mask.shape) == 2 and len(flowu.shape) == 2 and len(flowv.shape) == 2
        
        positions_to_zero = np.where(hand_mask)
        hand_mask_num = len(positions_to_zero[0])
        total_num = hand_mask.shape[0]*hand_mask.shape[1]
        
        flowu_copy = deepcopy(flowu)
        # print(torch.sum(flowu_copy) / total_num)
        flowu_copy[positions_to_zero] = 0
        
        flowv_copy = deepcopy(flowv)
        # print(torch.sum(flowv_copy) / total_num)
        flowv_copy[positions_to_zero] = 0
        
        target_mean = [torch.sum(flowu_copy)/(total_num-hand_mask_num), torch.sum(flowv_copy)/(total_num-hand_mask_num)]
        flowu[positions_to_zero] = target_mean[0]
        flowv[positions_to_zero] = target_mean[1]
        
        # print(f'flow shape: {flowu.shape}, {flowv.shape}')
        # print(torch.sum(flowu) / total_num)
        # print(torch.sum(flowv) / total_num)
        # print((torch.sum(flowu) - target_mean[0]*hand_mask_num) / (total_num - hand_mask_num))
        # print((torch.sum(flowv) - target_mean[1]*hand_mask_num) / (total_num - hand_mask_num))
        # print(target_mean)
        # print(f'hand_mask_num: {hand_mask_num}')
        # print('----------------')
        return Image.fromarray(flowu.cpu().numpy().astype(np.uint8), mode='P'), Image.fromarray(flowv.cpu().numpy().astype(np.uint8), mode='P')
        

