import datetime
from os import path
import math
# import git

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.trainer import XMemTrainer
# from dataset.static_dataset import StaticTransformDataset
# from dataset.vos_dataset import VOSDataset
from dataset.EPIC_dataset import EPICDataset
from dataset.EPIC_usetest_to_train import EPICTestToTrainDataset
from util.logger import TensorboardLogger
from util.configuration import Configuration
from util.load_subset import load_sub_davis, load_sub_yv
from argparse import ArgumentParser
from util.exp_handler import *
import pathlib
from visualize.visualize_eval_result_eps import visualize_eval_result
import wandb

def get_EPIC_parser():
    parser = ArgumentParser()

    # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
    parser.add_argument('--benchmark', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')

    # Data parameters
    parser.add_argument('--epic_root', help='EPIC data root', default='./EPIC_train') # TODO
    parser.add_argument('--yaml_root', help='yaml root', default='./EPIC_train/EPIC100_state_positive_train.yaml')
    parser.add_argument('--num_workers', help='Total number of dataloader workers across all GPUs processes', type=int, default=16)

    parser.add_argument('--key_dim', default=64, type=int)
    parser.add_argument('--value_dim', default=512, type=int)
    parser.add_argument('--hidden_dim', default=64, help='Set to =0 to disable', type=int)

    parser.add_argument('--deep_update_prob', default=0.2, type=float)

    """
    Stage-specific learning parameters
    Batch sizes are effective -- you don't have to scale them when you scale the number processes
    """
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--steps', nargs="*", default=[1000,4000], type=int)
    parser.add_argument('--lr', help='Initial learning rate', default=1e-5, type=float)
    parser.add_argument('--num_ref_frames', default=3, type=int)
    parser.add_argument('--num_frames', default=5, type=int)
    parser.add_argument('--start_warm', default=500, type=int)
    parser.add_argument('--end_warm', default=3000, type=int)

    parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)

    # Loading
    parser.add_argument('--load_network', help='Path to pretrained network weight only')
    parser.add_argument('--load_checkpoint', help='Path to the checkpoint file, including network, optimizer and such')

    # Logging information
    parser.add_argument('--log_text_interval', default=100, type=int)
    parser.add_argument('--log_image_interval', default=2500, type=int)
    parser.add_argument('--save_network_interval', default=500, type=int)
    parser.add_argument('--save_checkpoint_interval', default=15000, type=int)

    parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')
    parser.add_argument('--no_flow', help='without using flow information', action='store_true')
    parser.add_argument('--freeze', default=1, type=int, choices=[0,1])

    # # Multiprocessing parameters, not set by users
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')
    parser.add_argument('--en_wandb', action='store_true')
    parser.add_argument('--use_dice_align', action='store_true')
    parser.add_argument('--cos_lr', action='store_true')
    parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return {**vars(args), **{'amp': not args.no_amp}, **{'use_flow': not args.no_flow}}

"""
Initial setup
"""
# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
# 只针对于EPIC数据集
config = get_EPIC_parser()

# if config['benchmark']:
torch.backends.cudnn.benchmark = True

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'I am rank {local_rank} in this world of size {world_size}!')

# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

config['single_object'] = False

config['num_gpus'] = world_size
if config['batch_size']//config['num_gpus']*config['num_gpus'] != config['batch_size']:
    raise ValueError('Batch size must be divisible by the number of GPUs.')
# 分配给每个GPU，每个GPU的bs、worker
config['batch_size'] //= config['num_gpus']
config['num_workers'] //= config['num_gpus']
print(f'We are assuming {config["num_gpus"]} GPUs.')

print(f'We are now starting stage EPIC')

if config['debug']:
    config['batch_size'] = 1
    config['num_frames'] = 3
    config['iterations'] = 3
    config['finetune'] = 0
    config['log_text_interval'] = config['log_image_interval'] = 1
    config['save_network_interval'] = config['save_checkpoint_interval'] = 2
    config['log_image_interval'] = 1

"""
Model related
"""
if local_rank == 0:    
    # exp_handler
    exp = ExpHandler(en_wandb=config['en_wandb'], resume=config['resume'])
    exp.save_config(config)
    wandb.define_metric('eval_step')
    # Construct the rank 0 model
    model = XMemTrainer(config, logger=exp, 
                    save_path=exp._save_dir, 
                    local_rank=local_rank, world_size=world_size).train()
else:
    # Construct model for other ranks
    exp = None
    model = XMemTrainer(config, local_rank=local_rank, world_size=world_size).train()

# Load pertrained model if needed
if config['load_checkpoint'] is not None:
    total_iter = model.load_checkpoint(config['load_checkpoint'])
    config['load_checkpoint'] = None
    print('Previously trained model loaded!')
else:
    total_iter = 0

if config['load_network'] is not None:
    print('I am loading network from a disk, as listed in configuration')
    model.load_network(config['load_network'])
    config['load_network'] = None

    
"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id): 
    worker_seed = torch.initial_seed()%(2**31) + worker_id + local_rank*100
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def construct_loader(dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'],
                            worker_init_fn=worker_init_fn, drop_last=True)
    return train_sampler, train_loader

# BL 30K数据集
def renew_epic_loader(max_skip, finetune=False):
    #TODO for debug
    # train_dataset = EPICDataset(config['epic_root'], config['yaml_root'], max_skip, num_frames=config['num_frames'], finetune=finetune)
    train_dataset = EPICTestToTrainDataset(config['epic_root'], config['yaml_root'], max_skip, num_frames=config['num_frames'], finetune=finetune)
    print(f'EPIC dataset size: {len(train_dataset)}')
    print(f'Renewed with max_skip = {max_skip}')

    return construct_loader(train_dataset)

"""
Dataset related
"""

"""
These define the training schedule of the distance between frames
We will switch to max_skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
Not effective for stage 0 training
The initial value is not listed here but in renew_vos_loader(X)
"""
# 每隔多少帧进行一次采样，curriculum learning
# DAVIS帧率是24，EPIC是60，算了用6倍间隔
max_skip_values = [60, 90, 40, 40]

# 在训练的第10%，30%，80%的时候change max skip_values
increase_skip_fraction = [0.1, 0.3, 0.8, 100]

train_sampler, train_loader = renew_epic_loader(max_skip_values[0])
renew_loader = renew_epic_loader


"""
Determine max epoch
"""
total_epoch = math.ceil(config['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print(f'We approximately use {total_epoch} epochs.')

change_skip_iter = [round(config['iterations']*f) for f in increase_skip_fraction]
# Skip will only change after an epoch, not in the middle
print(f'The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}')

"""
Starts training
"""
finetuning = False
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)
if not config["only_eval"]:
    try:
        while total_iter < config['iterations'] + config['finetune']:
            
            # Crucial for randomness! 
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')

            # Train loop
            model.train()
            for data in train_loader:
                # Update skip if needed
                # 达到change skip iter之后，更新skip，右移change_skip_iter，并重新获取数据集
                # skip是每次跳过多少帧
                if total_iter >= change_skip_iter[0]:
                    while total_iter >= change_skip_iter[0]:
                        cur_skip = max_skip_values[0]
                        max_skip_values = max_skip_values[1:]
                        change_skip_iter = change_skip_iter[1:]
                    print(f'Changing skip to {cur_skip=}')
                    train_sampler, train_loader = renew_loader(cur_skip)
                    break

                # fine-tune means fewer augmentations to train the sensory memory
                # finetuning
                if config['finetune'] > 0 and not finetuning and total_iter >= config['iterations']:
                    train_sampler, train_loader = renew_loader(cur_skip, finetune=True)
                    finetuning = True
                    model.save_network_interval = 1000
                    break

                model.do_pass(data, total_iter)
                total_iter += 1

                if total_iter >= config['iterations'] + config['finetune']:
                    break
    finally:
        # not config['debug'] and total_iter>5000
        if model.logger is not None:
            model.save_network(total_iter)
            model.save_checkpoint(total_iter)
        
if local_rank == 0 and exp is not None:
    eval_iters = (config['iterations'] + config['finetune'])//config['save_network_interval']
    eval_iters = [it*config['save_network_interval'] for it in range(1, eval_iters+1)]
    if total_iter != eval_iters[-1]:
        eval_iters.append(total_iter)
        
    for iteration in eval_iters:
        exp_name = os.getenv('exp_name', default='default_group')
        home = pathlib.Path.home()
        wandb_project = os.getenv('WANDB_PROJECT', default='default_project')
        model_path = f'{home}/.exp/{wandb_project}/{exp_name}/{exp._exp_id}/network_{iteration}.pth'
        if not os.path.exists(model_path):
            print(f'Model not found: {model_path}')
            continue
        output_path = f'{home}/.exp/{wandb_project}/{exp_name}/{exp._exp_id}/eval_{iteration}'
        os.makedirs(output_path, exist_ok=True)
        os.system(f'python eval_EPIC.py --model "{model_path}" --output "{output_path}"')
        os.chdir('./XMem_evaluation')
        os.system(f'python evaluation_method.py --results_path "{output_path}"')
        os.chdir('..')
        os.system(f'zip -qru {output_path}/masks.zip {output_path}/')
        os.system(f'rm -r {output_path}/P*')
        os.system(f'rm -r "{output_path}/draw"')
    run_dir = f'{home}/.exp/{wandb_project}/{exp_name}/{exp._exp_id}'
    iters, JF_list = visualize_eval_result(run_dir)
    for i in range(len(iters)):
        exp.log_eval_acc(JF_list[i], iters[i])
    # exp.write
# network_in_memory = model.XMem.module.state_dict()

distributed.destroy_process_group()
