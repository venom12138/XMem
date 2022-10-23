from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import glob
import os
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np 
from skimage.io import imsave
import pdb

parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py', type=str)
parser.add_argument("--checkpoint_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/best_mIoU_iter_42000.pth', type=str)
parser.add_argument("--img_dir", default='../data/train/image', type=str)
parser.add_argument("--pred_seg_dir", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/outputs/train_seg', type=str)
args = parser.parse_args()

os.makedirs(args.pred_seg_dir, exist_ok = True)

# build the model from a config file and a checkpoint file
model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda')
palette = [0,0,0,255,255,255,128,0,0,0,128,0,0,0,128,255,0,0,255,255,0]
others = list(np.random.randint(0,255,size=256*3-len(palette)))
palette.extend(others)

alpha = 0.5
for file in tqdm(glob.glob(args.img_dir + '/*')):
    fname = os.path.basename(file).split('.')[0]
    # img = np.array(Image.open(os.path.join(args.img_dir, fname + '.jpg')))
    seg_result = inference_segmentor(model, file)[0]
    # print(type(seg_result))
    # print(seg_result.shape)
    seg_result = Image.fromarray(seg_result.astype(np.uint8)).convert('P')
    seg_result.putpalette(palette)
    seg_result.save(os.path.join(args.pred_seg_dir, fname + '.png'))
    # imsave(os.path.join(args.pred_seg_dir, fname + '.png'), seg_result.astype(np.uint8))



