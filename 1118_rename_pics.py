import os
from PIL import Image
from glob import glob


images_dir = '/home/venom/projects/XMem/visuals/P01/1118_baseline_noflow_draw/P01_14/P01_14_343'
uid = images_dir.split('/')[-1]
output_dir = f'./visuals/supplementary_video/baseline/{uid}'

os.makedirs(output_dir,exist_ok=True)
for pic in sorted(glob(images_dir + '/*.jpg')):
    img = Image.open(pic)
    index = int(pic.split('/')[-1].split('.')[0].split('_')[-1])
    img.save(f"{output_dir}/{str(index)}.jpg")