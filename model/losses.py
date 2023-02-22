import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import numpy as np



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

# dicrete dice loss # take in logits
# mask：B x (maxobj) x H x W
# tgt_mask: B x (maxobj+1) x H x W
def dice_loss_between_mask(mask1, tgt_mask): 
    num_classes = mask1.shape[1]
    tgt_mask = torch.argmax(tgt_mask, dim=1)
    # print(f'tgt_mask:{tgt_mask.shape}')
    tgt_mask = F.one_hot(tgt_mask, num_classes=num_classes+1).permute(0, 3, 1, 2).float()
    # print(f'tgt_mask:{tgt_mask.shape}')
    mask1 = mask1.flatten(start_dim=2) # B x maxobj x HW
    tgt_mask = tgt_mask.flatten(start_dim=2) # B x (maxobj+1) x HW
    tgt_mask = tgt_mask[:,1:,:] # B x maxobj x HW
    # print(f'tgt_mask:{tgt_mask.shape}')
    
    numerator = 2 * (mask1 * tgt_mask).sum(-1)
    # print(f'mask1 * tgt_mask:{(mask1 * tgt_mask).shape}')
    # print(f'numerator:{numerator}')
    
    denominator = mask1.sum(-1) + tgt_mask.sum(-1)
    # print(f'denominator:{denominator}')
    loss = 1 - (numerator + 1) / (denominator + 1)
    # print(loss)
    # dd
    return loss.mean()

#continual dice loss take in mask
# def dice_loss_between_mask(mask1, mask2): # mask：B x maxobj x H x W
    
#     mask1 = mask1.flatten(start_dim=2) # B x (maxobj) x HW
#     mask2 = mask2.flatten(start_dim=2) # B x (maxobj) x HW

#     numerator = 2 * (mask1 * mask2).sum(-1)
    
#     denominator = mask1.sum(-1) + mask2.sum(-1)
    
#     loss = 1 - (numerator + 1) / (denominator + 1)
    
#     return loss.mean()

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
        # self.bkl = BootstrappedKL(config['start_warm'], config['end_warm'])
        
        self.start_w = 1
        self.end_w = 0.0
        
    def compute(self, data, num_objects, it):
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]
        # print(t)
        losses['total_loss'] = 0
        for ti in range(0, t):
            # weight = self.end_w + (self.start_w - self.end_w) * (ti / t)
            # weight = torch.tensor(weight, dtype=torch.float).cuda()
            # weight.requires_grad=False
            weight = self.end_w + (self.start_w - self.end_w) * (ti / (t-1))
            for bi in range(b):
                if ti == t-1:
                    loss, p = self.bce(data[f'flogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,1,0], it)
                    losses['p'] += p / b / 2 # (t-1)
                    losses[f'ce_loss_{ti}'] += loss / b
                elif ti == 0:
                    loss, p = self.bce(data[f'blogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,0,0], it)
                    losses['p'] += p / b / 2 # (t-1)
                    losses[f'ce_loss_{ti}'] += loss / b
                # else:
                #     if it%2 == 0:
                #         loss, p = self.bkl(data[f'flogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data[f'blogits_{ti}'][bi:bi+1, :num_objects[bi]+1], it) # 这里是把有objects的给拿出来了，附带上一个背景
                #         loss = loss * weight
                #     else:
                #         loss, p = self.bkl(data[f'blogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data[f'flogits_{ti}'][bi:bi+1, :num_objects[bi]+1], it)
                #         loss = loss * (1-weight+self.end_w)
                

            losses['total_loss'] += losses['ce_loss_%d'%ti]
            
            if ti == 0:
                losses[f'dice_loss_{ti}'] = dice_loss(data[f'bmasks_{ti}'], data['cls_gt'][:,0,0]) # dice loss评估相似性 X交Y/X+Y
                losses['total_loss'] += losses[f'dice_loss_{ti}']
            elif ti == t - 1:
                losses[f'dice_loss_{ti}'] = dice_loss(data[f'fmasks_{ti}'], data['cls_gt'][:,1,0]) # dice loss评估相似性 X交Y/X+Y
                losses['total_loss'] += losses[f'dice_loss_{ti}']
            else:
                if self.config['use_dice_align']:
                    if np.random.rand() < weight:
                        losses[f'dice_loss_{ti}'] = dice_loss_between_mask(data[f'fmasks_{ti}'], data[f'blogits_{ti}'].detach())
                    else:
                        losses[f'dice_loss_{ti}'] = dice_loss_between_mask(data[f'bmasks_{ti}'], data[f'flogits_{ti}'].detach())
                    losses['total_loss'] += losses[f'dice_loss_{ti}']
                if it >= self.config['teacher_warmup']:
                    if self.config['use_teacher_model']:
                        losses[f'sftf_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'fmasks_{ti}'], data[f't_flogits_{ti}'].detach())
                        losses[f'sbtb_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'bmasks_{ti}'], data[f't_blogits_{ti}'].detach())
                        losses['total_loss'] += losses[f'sftf_dice_loss_{ti}'] + losses[f'sbtb_dice_loss_{ti}']
                        if self.config['ts_all_align_loss']:
                            losses[f'sftb_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'fmasks_{ti}'], data[f't_blogits_{ti}'].detach())
                            losses[f'sbtf_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'bmasks_{ti}'], data[f't_flogits_{ti}'].detach())
                            losses['total_loss'] += losses[f'sftb_dice_loss_{ti}'] + losses[f'sbtf_dice_loss_{ti}']
        return losses


# class Random_Walk_Loss:
#     def __init__(self, dropout_rate=0.0, temperature=0.07):
#         self.dropout_rate = dropout_rate
#         self.temperature = temperature
#         self._xent_targets = dict()
#         self.xent = nn.CrossEntropyLoss(reduction="none")
        
#     def cal_affinity(self, x1, x2):
#         in_t_dim = x1.ndim
#         if in_t_dim < 4:  # add in time dimension if not there
#             x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

#         A = torch.einsum('bctn,bctm->btnm', x1, x2)

#         return A.squeeze(1) if in_t_dim < 4 else A

#     def stoch_mat(self, A, do_dropout=True):
#         ''' Affinity -> Stochastic Matrix '''

#         if do_dropout and self.dropout_rate > 0:
#             A[torch.rand_like(A) < self.dropout_rate] = -1e20

#         return F.softmax(A/self.temperature, dim=-1)

#     def xent_targets(self, A):
#         B, N = A.shape[:2]
#         key = '%s:%sx%s' % (str(A.device), B,N)

#         if key not in self._xent_targets:
#             I = torch.arange(A.shape[-1])[None].repeat(B, 1)
#             self._xent_targets[key] = I.view(-1).to(A.device)

#         return self._xent_targets[key]
        
#     def compute_random_walk_loss(self, keys):
#         # keys: B, key_dim, num_frames, H//16, W//16
#         keys = keys.view(keys.shape[0],keys.shape[1],keys.shape[2],-1) # B, C, T, N
#         keys = F.normalize(keys, p=2, dim=1)
#         B, C, T, N = keys.shape
#         walks = dict()
#         As = self.cal_affinity(keys[:, :, :-1], keys[:, :, 1:])
#         A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)]
        
#         A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)]
#         AAs = []
#         for i in list(range(1, len(A12s))): # len就是T
#             g = A12s[:i+1] + A21s[:i+1][::-1]
#             aar = g[0]
#             for _a in g[1:]:
#                 aar = aar @ _a
#             AAs.append((f"r{i}", aar))

#         for i, aa in AAs:
#             walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]
        
#         # Compute loss 
#         #################################################################
#         xents = 0
#         diags = dict()

#         for name, (A, target) in walks.items():
#             logits = torch.log(A+EPS).flatten(0, -2)
#             loss = self.xent(logits, target).mean()
#             acc = (torch.argmax(logits, dim=-1) == target).float().mean()
#             diags.update({f"rand_walk/xent_{name}": loss.detach(),
#                             f"rand_walk/acc_{name}": acc})
#             xents += loss
        
#         return loss, diags

if __name__ == '__main__':
    keys = torch.rand(2, 256, 8, 24, 24)
    rand_loss = Random_Walk_Loss()
    loss, diags = rand_loss.compute_random_walk_loss(keys)
    print(loss)
    print(diags)