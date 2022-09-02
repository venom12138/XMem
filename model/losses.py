import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict


def dice_loss(input_mask, cls_gt): # cls_gt is B x H x W
    num_objects = input_mask.shape[1] # input_mask is B x max_obj_num x H x W
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i].flatten(start_dim=1) # B x HW
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt==(i+1)).float().flatten(start_dim=1) # B x HW
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            # 这个结果就是/384/384之后的
            # 相当于是每一个pixel的概率之间算了cross entropy，最后除以pixel的数量
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1) # raw_loss [H*W]
        num_pixels = raw_loss.numel() # num_pixels = H*W

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class BootstrappedKL(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
    
    def forward(self, input, target, it):
        if it < self.start_warm:
            # pixel wise的平均值
            # [TODO]:input 和 target detach掉一个， 
            return F.kl_div(input.softmax(dim=1).log(), target.detach().softmax(dim=1), reduction='sum')/input.shape[-1]/input.shape[-2], 1.0
        # raw_loss: [(num_objects+1)*H*W]
        # raw_loss应该是每一个pixel一个的
        raw_loss = F.kl_div(input.softmax(dim=1).log(), target.detach().softmax(dim=1), reduction='none').sum(dim=1).view(-1) # /input.shape[0]
        num_pixels = raw_loss.numel() 

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])
        self.bkl = BootstrappedKL(config['start_warm'], config['end_warm'])
        self.start_w = 1
        self.end_w = 0.2

    def compute(self, data, num_objects, it):
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]
        # print(t)
        losses['total_loss'] = 0
        for ti in range(0, t):
            weight = self.end_w + (self.start_w - self.end_w) * (ti / t)
            for bi in range(b):
                if ti == t-1:
                    loss, p = self.bce(data[f'flogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,1,0], it)
                elif ti == 0:
                    loss, p = self.bce(data[f'blogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,0,0], it)
                else:
                    if it%2 == 0:
                        loss, p = weight * self.bkl(data[f'flogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data[f'blogits_{ti}'][bi:bi+1, :num_objects[bi]+1], it) # 这里是把有objects的给拿出来了，附带上一个背景
                    else:
                        loss, p = (1-weight+self.end_w) * self.bkl(data[f'blogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data[f'flogits_{ti}'][bi:bi+1, :num_objects[bi]+1], it)
                losses['p'] += p / b / (t-1)
                losses[f'ce_loss_{ti}'] += loss / b

            losses['total_loss'] += losses['ce_loss_%d'%ti]
            # TODO: 中间帧做dice loss
            if ti == 0:
                losses[f'dice_loss_{ti}'] = dice_loss(data[f'bmasks_{ti}'], data['cls_gt'][:,0,0]) # dice loss评估相似性 X交Y/X+Y
                losses['total_loss'] += losses[f'dice_loss_{ti}']
            elif ti == t - 1:
                losses[f'dice_loss_{ti}'] = dice_loss(data[f'fmasks_{ti}'], data['cls_gt'][:,1,0]) # dice loss评估相似性 X交Y/X+Y
                losses['total_loss'] += losses[f'dice_loss_{ti}']
            else:
                losses[f'dice_loss_{ti}'] = dice_loss(data[f'fmasks_{ti}'], data[f'bmasks_{ti}'])
                losses['total_loss'] += losses[f'dice_loss_{ti}']

        return losses
