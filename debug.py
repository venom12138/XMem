import cv2
import numpy as np
import glob
import os
import time
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image
import matplotlib.pyplot as plt
import seaborn
# img = cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/myflow_frames/P01_01/37/u/frame_0000007029.jpg')

img0 = np.array(cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/positive_frames/P01_01/37/frame_0000007034.jpg'), dtype =np.int32)
img1 = np.array(cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/positive_frames/P01_01/37/frame_0000007035.jpg'), dtype =np.int32)
img2 = np.array(cv2.imread('/cluster/home2/yjw/venom/XMem/data/P01/positive_frames/P01_01/37/frame_0000007036.jpg'), dtype =np.int32)
plt.figure()
ax = seaborn.heatmap(np.sum(np.abs(img0 - img1), axis=2), cmap='coolwarm', vmin=0, vmax=1, )
ax.get_figure().savefig('temp1.jpg')
plt.clf()

plt.figure()
ax = seaborn.heatmap(np.sum(np.abs(img2 - img1), axis=2), cmap='coolwarm', vmin=0, vmax=1, )
ax.get_figure().savefig('temp2.jpg')
plt.clf()
print(np.max(np.abs(img0-img1)))
print(np.max(np.abs(img2-img1)))
print(img0[0,0])

