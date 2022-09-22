import os
from os import path
from argparse import ArgumentParser
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from dataset.EPIC_testdataset import EPICtestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar
from tqdm import tqdm
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

# try:
#     import hickle as hkl
# except ImportError:
#     print('Failed to import hickle. Fine if not using multi-scale testing.')

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    # h,w = new_mask.size
    palette = [0,0,0,255,255,255,128,0,0,0,128,0,0,0,128,255,0,0,255,255,0]
    others = list(np.random.randint(0,255,size=256*3-len(palette)))
    palette.extend(others)
    new_mask.putpalette(palette)

    return new_mask
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='/home/venom/projects/XMem/saves/XMem.pth')

# Data options
parser.add_argument('--EPIC_path', default='./val_data')
parser.add_argument('--yaml_path', default='./val_data/EPIC100_state_positive_val.yaml')
parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G', default='EPIC')
parser.add_argument('--split', help='val/test', default='val')
parser.add_argument('--output', default=None)
parser.add_argument('--save_all', action='store_true', 
            help='Save all frames. Useful only in YouTubeVOS/long-time video', )

parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
        
# Long-term memory options
parser.add_argument('--disable_long_term', action='store_true')
parser.add_argument('--not_use_flow', action='store_true')
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

# Multi-scale options
parser.add_argument('--save_scores', action='store_true')

args = parser.parse_args()

config = vars(args)
config['enable_long_term'] = not config['disable_long_term']
config['enable_long_term_count_usage'] = True

if args.output is None:
    # args.output = f'./output/{args.dataset}_{args.split}'
    args.output = f"./output/{args.model.split('/')[-1][:-4]}"
    print(f'Output path not provided. Defaulting to {args.output}')

"""
Data preparation
"""
out_path = args.output
print(out_path)
use_flow = not args.not_use_flow
if 'noflow' in args.model:
    use_flow = False
if use_flow == False:    
    print('not use flow !!!!!!!!!!!')

dataset = EPICtestDataset(args.EPIC_path, args.yaml_path)
val_loader = DataLoader(dataset, 1,  shuffle=False, num_workers=2)
torch.autograd.set_grad_enabled(False)

# Load our checkpoint
network = XMem(config, args.model).cuda().eval()

# if args.model is not None:
#     model_weights = torch.load(args.model)
#     network.load_weights(model_weights, init_as_zero_if_needed=True)
# else:
#     print('No model loaded.')

total_process_time = 0
total_frames = 0

# Start eval
for data in tqdm(val_loader):
    vid_name = data['info']['name'][0]
    vid_length = data['rgb'][0].shape[0]
    # no need to count usage for LT if the video is not that long anyway
    # config['enable_long_term_count_usage'] = (
    #     config['enable_long_term'] and
    #     (vid_length
    #         / (config['max_mid_term_frames']-config['min_mid_term_frames'])
    #         * config['num_prototypes'])
    #     >= config['max_long_term_elements']
    # )

    # mapper = MaskMapper()
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False

    for ti in (range(vid_length)):
        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            rgb = data['rgb'][0][ti].cuda() # 3*H*W
            flow = data['flow'][0][ti].cuda() # 2*H*W
            if ti == 0:
                msk = data['first_frame_gt'][0][ti].cuda() # H*W
                num_objects = msk.shape[0]
                processor.set_all_labels(range(1, num_objects+1))
            else:
                msk = None
            frame = data['info']['frames'][ti][0]
            shape = dataset.img_size # H*W
            whether_to_save_mask = int(data['whether_save_mask'][0][ti].cpu())
            need_resize = True
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue

            # if args.flip:
            #     rgb = torch.flip(rgb, dims=[-1])
            #     msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            # Map possibly non-continuous labels to continuous ones
            # if msk is not None:
                # msk, labels = mapper.convert_mask(msk[0].numpy())# convert to one hot
                # msk = torch.Tensor(msk).cuda()
                # if need_resize:
                #     msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
            #     processor.set_all_labels(list(mapper.remappings.values()))
            # else:
            #     labels = None

            # Run the model on this frame
            if msk is not None:
                if use_flow:
                    prob = processor.step(rgb, flow, msk, end=(ti==vid_length-1))
                else:
                    prob = processor.step(rgb, flow=None, mask = msk, end=(ti==vid_length-1))
            else:
                if use_flow:
                    prob = processor.step(rgb, flow, end=(ti==vid_length-1))
                else:
                    prob = processor.step(rgb, flow=None, end=(ti==vid_length-1))

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end)/1000)
            total_frames += 1

            # if args.flip:
            #     prob = torch.flip(prob, dims=[-1])

            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            # print(out_mask.shape)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
            # print(out_mask)
            # dd
            # if args.save_scores:
            #     prob = (prob.detach().cpu().numpy()*255).astype(np.uint8)

            # Save the mask
            # print(whether_to_save_mask)
            if (args.save_all or whether_to_save_mask) and msk is None:
                # print('save')
                partition = vid_name.split('_')[0]
                video_part = '_'.join(vid_name.split('_')[:2])
                this_out_path = path.join(out_path, partition, video_part, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = colorize_mask(out_mask)
                out_mask.save(os.path.join(this_out_path, frame.replace('jpg','png')))
                # out_mask = mapper.remap_index_mask(out_mask)
                # out_img = Image.fromarray(out_mask)
                # plt.imsave(os.path.join(this_out_path, frame.replace('jpg','png')), out_mask*255, cmap='gray')
                # out_img.save(os.path.join(this_out_path, frame))

            # if args.save_scores:
            #     np_path = path.join(args.output, 'Scores', vid_name)
            #     os.makedirs(np_path, exist_ok=True)
            #     if ti==len(loader)-1:
            #         hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
            #     if args.save_all or info['save'][0]:
            #         hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')


print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

# if not args.save_scores:
#     if is_youtube:
#         print('Making zip for YouTubeVOS...')
#         shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output, 'Annotations')
#     elif is_davis and args.split == 'test':
#         print('Making zip for DAVIS test-dev...')
#         shutil.make_archive(args.output, 'zip', args.output)
