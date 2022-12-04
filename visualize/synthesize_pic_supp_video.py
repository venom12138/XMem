import os
from PIL import Image
import cv2
import numpy as np
from glob import glob
import sys
from matplotlib import pyplot as plt
img_dir1 = '/home/venom/projects/XMem/visuals/P01/1118_baseline_noflow_draw/P01_14/P01_14_343'
img_dir2 = '/home/venom/projects/XMem/visuals/P01/1118noflow_draw/P01_14/P01_14_343'

img_dir3 = '/home/venom/projects/XMem/visuals/P22/1118_baseline_noflow_draw/P22_01/P22_01_164'
img_dir4 = '/home/venom/projects/XMem/visuals/P22/1118noflow_draw/P22_01/P22_01_164'

img_dir5 = '/home/venom/projects/XMem/visuals/P28/1118_baseline_noflow_draw/P28_25/P28_25_30'
img_dir6 = '/home/venom/projects/XMem/visuals/P28/1118noflow_draw/P28_25/P28_25_30'

max_length = 0

for i in range(1, 7):
    cur_img_dir = eval(f"img_dir{i}")
    max_length = max(len(glob(f"{cur_img_dir}/*.jpg")), max_length)
    cur_imgs = list(sorted(glob(f'{cur_img_dir}/*.jpg')))
    exec(f"all_imgs_{i} = cur_imgs")
    
print(f'max_length: {max_length}')
w = 256
h = 456
font = cv2.FONT_HERSHEY_SIMPLEX
for j in range(max_length):
    print(f'j:{j}')
    output_img = np.zeros((2*w, h*3+150+20, 3), dtype=np.uint8)
    cv2.putText(output_img, 'Baseline', (5, w//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(output_img, 'XMem+BAMT', (5, w//2 * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    # plt.imshow(output_img)
    # plt.show()
    
    for i in range(1, 7):
        all_imgs = eval(f"all_imgs_{i}")
        
        img_path = all_imgs[j%len(all_imgs)]
        
        img = Image.open(img_path)

        print(f'img_size: {np.array(img).shape}')
        img = np.array(img.resize((h, w)))
        
        col_cnt = (i+1)%2
        row_cnt = (i+1)//2 - 1
        print(f"image shape:{img.shape}")
        output_img[(col_cnt+0)*w:(col_cnt+1)*w,
                        row_cnt*h+10*row_cnt+150: (row_cnt+1)*h+150+10*row_cnt, :] = img

    Image.fromarray(output_img).save(f'../visuals/supplementary_video/synthesized/image{str(j).zfill(4)}.jpg')
