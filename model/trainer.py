"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import git
import datetime
# TODO change to relative path
sys.path.append('/home/venom/projects/XMem/')
print(sys.path)
# from model.network import XMem
# from model.losses import LossComputer
from network import XMem
from losses import LossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
from util.configuration import Configuration
from util.logger import TensorboardLogger
import resnet

class XMemTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank
        try:
            self.XMem = nn.parallel.DistributedDataParallel(
                XMem(config).cuda(), 
                device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        except:
            self.XMem = nn.parallel.DataParallel(
                XMem(config).cuda())
        # Set up logger when local_rank = 0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size', str(sum([param.nelement() for param in self.XMem.parameters()])))
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.XMem.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        # [b, num_frames, 3, H, W]
        frames = data['rgb']
        # [b, 1, max_num_obj, H, W]
        first_frame_gt = data['first_frame_gt'].float()
        b = frames.shape[0]
        # data['info']['num_objects']: [], len=b, 每一个数代表每一个clip的object数量
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        # 此处的num_objects是max_num_obj，而不是每一个clip的object数量
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            # image features never change, compute once
            # frames:[B,num_frames,C,H,W]
            # key:[B, key_dim, num_frames, H//16, W//16]
            # shrinkage:[B, num_frames, 1, H//16, W//16]
            # selection:[B, num_frames, key_dim, H//16, W//16]
            # f16:[B, num_frames, 1024, H//16, W//16]
            # f8:[B, num_frames, 512, H//8, W//8]
            # f4:[B, num_frames, 256, H//4, W//4]
            key, shrinkage, selection, f16, f8, f4 = self.XMem('encode_key', frames)

            filler_one = torch.zeros(1, dtype=torch.int64)
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            # first_frame_gt[:,0]:[b, max_num_obj, H, W]
            # hidden: [b, max_num_obj, hidden_dim, H, W]
            # f16[:,0]: [b, 1, 1024, H//16, W//16]
            # frames[:,0]: [b, 3, H, W]
            # encode_value只对采样的clip中的第一个frame和mask进行计算
            # v16:[b, max_obj_num, value_dim, H//16, W//16]
            # hidden:[b, max_obj_num, hidden_dim, H//16, W//16]
            v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
            # values:[b, max_obj_num, value_dim, 1, H//16, W//16]
            values = v16.unsqueeze(3) # add the time dimension
            
            for ti in range(1, self.num_frames):
                # 从memory 中 选取的ref_frame
                if ti <= self.num_ref_frames:
                    ref_values = values
                    # 取前ti个frame的key作为ref_key
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would 
                    # need broadcasting in gather which we don't have
                    indices = [
                        torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                    for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None

                # Segment frame ti
                memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                        ref_keys, ref_shrinkage, ref_values)
                # hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, 
                #         hidden, selector, h_out=(ti < (self.num_frames-1)))
                args = [(f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, hidden, selector]
                kwargs = {'h_out':(ti < (self.num_frames-1))} # 最后一帧不进行update
                # logits:B,max_obj_num+1,H,W; 
                # masks:B,max_obj_num,H,W
                # hidden:B,max_obj_num,hidden_dim,H//16,W//16
                hidden, logits, masks = self.XMem('segment', *args, **kwargs) 

                # No need to encode the last frame
                if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob # 50%的概率进行deep update
                    v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3) # 更新后的value用于下一帧，也就是每一帧的alue都是用的上一帧的

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, num_filled_objects), it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward() 
            self.optimizer.step()

        self.scheduler.step()

    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.XMem.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
        checkpoint = { 
            'it': it,
            'network': self.XMem.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.XMem.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.XMem.module.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # 不使用BN和dropout
        self.XMem.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.XMem.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.XMem.eval()
        return self

if __name__ == '__main__':
    raw_config = Configuration()
    raw_config.parse()
    stage_config = raw_config.get_stage_parameters('0')
    config = dict(**raw_config.args, **stage_config)
    # repo = git.Repo("..")
    # git_info = str(repo.active_branch)+' '+str(repo.head.commit.hexsha)
    if config['exp_id'].lower() != 'null':
        print('I will take the role of logging!')
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'])
    else:
        long_id = None
    logger = TensorboardLogger(config['exp_id'], long_id, '123456')
    logger.log_string('hyperpara', str(config))
    data = {
            'rgb': torch.rand(2,3,3,384,384), # [b, num_frames, 3, H, W]
            'first_frame_gt': torch.randint(0, 1, (2,1,5,384,384)), # [b, 1, max_num_obj, H, W] one hot
            'cls_gt': torch.randint(0,2,(2,3,1,384,384)), # [b, num_frames, 1, H, W]
            'selector': torch.tensor([[1, 1, 1, 0, 0],[1, 1, 1, 0, 0]]), # [b,max_num_obj] 前num_objects个是1，后面是0
            'info': {'num_frames': torch.tensor([3,3]), 'num_objects': torch.tensor([3,3])},
        }
    model = XMemTrainer(config).train()
    network = resnet.resnet50(pretrained=False)
    print(model.XMem)
    print(f'----------')
    print(network)
    model.do_pass(data, 1)