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

class DataPreprocess():
    def __init__(self, config_file='saves/seg_twohands_ccda/seg_twohands_ccda.py', \
                checkpoint_file='saves/seg_twohands_ccda/best_mIoU_iter_56000.pth', \
                num_frames=3, max_num_obj=3, remove_hand=True, finetune=False):
        self.remove_hand = remove_hand
        self.max_num_obj = max_num_obj
        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(config_file, checkpoint_file, device=f'cuda')

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])
        # 仿射变换：平移旋转之类的
        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])
        
        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
        ])
        
        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
    
    def preprocess(self, data):
        # images:[B, num_frames, H, W, 3]
        images = data['rgb'] 
        # flows:[B, num_frames, 5, H, W, 2]: 2 for u and v, 5 for 5 neighbor flow
        flows = data['flows']
        # masks:[B, 2, H, W]: 2 for first and last
        masks = data['masks']
        
        B = images.shape[0]
        num_frames = images.shape[1]
        H = 384
        W = 384
        
        new_rgbs = torch.zeros((B, num_frames, 3, H, W), dtype=torch.float32)
        new_forward_flows = torch.zeros((B, num_frames, 10, H, W), dtype=torch.float32)
        new_backward_flows = torch.zeros((B, num_frames, 10, H, W), dtype=torch.float32)
        new_first_last_frame_gt = torch.zeros((B, 2, self.max_num_obj, H, W), dtype=torch.float32)
        new_cls_gt = torch.zeros((B, 2, 1, H, W), dtype=torch.float32)
        
        for bi in range(B):
            target_objects = data['target_objects'][bi]
            sequence_seed = np.random.randint(2147483647)
            pairwise_seed = np.random.randint(2147483647)
            video_mask = []
            for fi in range(num_frames):
                img = Image.fromarray(images[bi, fi].cpu().numpy(), mode='RGB')
                agg_flow = flows[bi, fi] # [5, H, W, 2]
                this_im = self.img_preprocess(img, sequence_seed, pairwise_seed)
                # forward_flow: [10, H, W]
                this_forward_flow, this_backward_flow = self.flow_preprocess(img, agg_flow, sequence_seed, pairwise_seed)
                if fi == 0:
                    this_gt = masks[bi, 0]
                    this_gt = self.mask_preprocess(this_gt, sequence_seed, pairwise_seed)
                    video_mask.append(this_gt)
                elif fi == num_frames - 1:
                    this_gt = masks[bi, 1]
                    this_gt = self.mask_preprocess(this_gt, sequence_seed, pairwise_seed)
                    video_mask.append(this_gt)
                
                new_rgbs[bi, fi]  = this_im
                new_forward_flows[bi, fi] = this_forward_flow
                new_backward_flows[bi, fi] = this_backward_flow
                
            # masks是一个[2, H, W]的np array
            video_mask = np.stack(video_mask, 0)
            # Generate one-hot ground-truth
            cls_gt = np.zeros((2, 384, 384), dtype=np.long) # 只有两帧有mask
            first_last_frame_gt = np.zeros((2, self.max_num_obj, 384, 384), dtype=np.long)
            
            # Image.fromarray(video_mask[0]).show()
            # dd
            
            # target_objects是一个list，长度是objects的数量
            for i, l in enumerate(target_objects):
                # masks是一个[2, H, W]的np array
                # this_mask一个[2, H, W]的np array, 其中每个像素值是true or false
                this_mask = (video_mask==l)
                # print(this_mask)
                # cls_gt是一个[2, H, W]的np array，将cls_gt和this_mask对应的位置赋上值
                try:
                    cls_gt[this_mask] = i+1
                except:
                    print(cls_gt.shape)
                    print(this_mask.shape)
                    print(masks.shape)
                    print(l)
                    print(i)
                    raise Exception('error')
                # first_frame_gt是一个one hot向量，[1, num_objects, H, W]
                # 将所有的num_frame里面的第一个，也就是第一个frame，里的第i个object赋给first_frame_gt，也就是一个0,1的array
                first_last_frame_gt[:,i] = this_mask
            # expand完变成(2, 1, H, W)
            cls_gt = np.expand_dims(cls_gt, 1)
            
            new_first_last_frame_gt[bi] = torch.tensor(first_last_frame_gt)
            new_cls_gt[bi] = torch.tensor(cls_gt)
            
        new_data = {
            'rgb': new_rgbs.cuda(), # [B, num_frames, 3, H, W]
            'forward_flow': new_forward_flows.cuda(), # [B, num_frames, 10, H, W]
            'backward_flow': new_backward_flows.cuda(), # [B, num_frames, 10, H, W]
            'first_last_frame_gt': new_first_last_frame_gt.cuda().to(torch.long), # [B, 2, max_num_obj, H, W] one hot
            'cls_gt': new_cls_gt.cuda().to(torch.long), # [B, 2, 1, H, W]
            'selector': data['selector'], # [max_num_obj] 前num_objects个是1，后面是0
            'text': data['text'],
            'info': data['info'],
        }
        
        return new_data
        
    def mask_preprocess(self, mask, sequence_seed, pairwise_seed):
        this_gt = Image.fromarray(mask.cpu().numpy().astype(np.uint8), mode='P')
        reseed(sequence_seed)
        this_gt = self.all_gt_dual_transform(this_gt)
        reseed(pairwise_seed)
        this_gt = self.pair_gt_dual_transform(this_gt)
        return np.array(this_gt.convert('P'))

    def flow_preprocess(self, img, agg_flow, sequence_seed, pairwise_seed):
        forward_flow = []
        backward_flow = []
        for ni in range(5):
            flowu = agg_flow[ni, :, :, 0]
            flowv = agg_flow[ni, :, :, 1]
            
            flowu = Image.fromarray(flowu.cpu().numpy(), mode='P')
            flowv = Image.fromarray(flowv.cpu().numpy(), mode='P')
            backward_u = Image.fromarray(255 - np.array(flowu), mode='P')
            backward_v = Image.fromarray(255 - np.array(flowv), mode='P')
            if self.remove_hand:
                flowu, flowv = self.remove_hand_region(np.array(img), np.array(flowu), np.array(flowv))
                backward_u, backward_v = self.remove_hand_region(np.array(img), np.array(backward_u), np.array(backward_v))
            
            reseed(sequence_seed)
            flowu = self.all_gt_dual_transform(flowu)
            reseed(sequence_seed)
            backward_u = self.all_gt_dual_transform(backward_u)
            
            reseed(sequence_seed)
            flowv = self.all_gt_dual_transform(flowv)
            reseed(sequence_seed)
            backward_v = self.all_gt_dual_transform(backward_v)
            
            reseed(pairwise_seed)
            flowu = self.pair_gt_dual_transform(flowu)
            reseed(pairwise_seed)
            backward_u = self.pair_gt_dual_transform(backward_u)

            reseed(pairwise_seed)
            flowv = self.pair_gt_dual_transform(flowv)
            reseed(pairwise_seed)
            backward_v = self.pair_gt_dual_transform(backward_v)
            
            # 将0-255的像素值映射到0到1之间并中心化
            flowu = transforms.ToTensor()(flowu)
            flowv = transforms.ToTensor()(flowv)
            flowu = flowu - torch.mean(flowu)
            flowv = flowv - torch.mean(flowv)
            
            # 将0-255的像素值映射到0到1之间并中心化
            backward_u = transforms.ToTensor()(backward_u)
            backward_v = transforms.ToTensor()(backward_v)
            backward_u = backward_u - torch.mean(backward_u)
            backward_v = backward_v - torch.mean(backward_v)
            
            forward_flow.append(flowu.squeeze())
            forward_flow.append(flowv.squeeze())
            backward_flow.append(backward_u.squeeze())
            backward_flow.append(backward_v.squeeze())
            
        # [10, H, W]
        return torch.stack(forward_flow), torch.stack(backward_flow)
    
    def img_preprocess(self, img, sequence_seed, pairwise_seed):
        reseed(sequence_seed)
        this_im = self.all_im_dual_transform(img)
        this_im = self.all_im_lone_transform(this_im)
        reseed(pairwise_seed)
        this_im = self.pair_im_dual_transform(this_im)
        this_im = self.pair_im_lone_transform(this_im)
        this_im = self.final_im_transform(this_im)
        return this_im
    
    # img: nparray [HWC]
    # flow: nparray [HWC]
    # 将手的区域的flow置为0
    def remove_hand_region(self, img, flowu, flowv):
        # print(f'dddimg:{img.shape}')
        flowu = torch.tensor(flowu).to(torch.float64)
        flowv = torch.tensor(flowv).to(torch.float64)
        assert img.shape[2] == 3 and len(flowu.shape) == 2 and len(flowv.shape) == 2
        
        hand_mask = inference_segmentor(self.model, img)[0].astype(np.uint8)
        
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
        return Image.fromarray(flowu.cpu().numpy(), mode='P'), Image.fromarray(flowv.cpu().numpy(), mode='P')
        

