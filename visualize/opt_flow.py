import cv2
import numpy as np
import glob
import os
import time
from sklearn.cluster import KMeans, DBSCAN
from PIL import Image
import matplotlib.pyplot as plt


# colors = [[255,255,255],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[0,0,0]]
# flow_dir = './data/P01/flow_frames/P01_01'# './data/P01/myflow_frames/P01_01/37'
# u_dir = flow_dir + '/u'
# v_dir = flow_dir + '/v'

# n_clusters=3
# # n_clusters={n_clusters}
# save_path = f'./data/P01/group/37/mapped_n_clusters={n_clusters}/'
# flow_idx = int(np.ceil((float(7022) - 3) / 2))
# for i, u_img_path in enumerate(sorted(list(glob.glob(u_dir + '/*.jpg')))):
#     if i < flow_idx:
#         continue
#     # print(np.array(cv2.imread(u_img_path, 0)).shape)
#     v_img_path = u_img_path.replace('u','v')
#     u_frame = np.expand_dims(cv2.imread(u_img_path, 0),axis=2)
#     v_frame = np.expand_dims(cv2.imread(v_img_path, 0),axis=2)
#     flow_frame = np.concatenate((u_frame,v_frame),axis=2)
#     flow_frame = flow_frame.reshape(flow_frame.shape[0]*flow_frame.shape[1],2)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flow_frame)
#     # kmeans = DBSCAN(eps=0.1, min_samples=10).fit(flow_frame)
#     # print(np.where(kmeans.labels_==0)[0].shape)
#     cluster_map = kmeans.labels_.reshape(u_frame.shape[0], u_frame.shape[1]).astype(np.uint8)
#     mask = np.zeros([cluster_map.shape[0],cluster_map.shape[1],3], dtype=np.uint8)
#     for i in range(n_clusters):
#         mask[cluster_map==i] = colors[i]
#     if not os.path.isdir(save_path):
#         os.makedirs(save_path)
#     # np.savetxt('./temp_data/' + u_img_path.split('/')[-1].split('.')[0]+'.txt', mask)
#     cv2.imwrite(save_path + u_img_path.split('/')[-1], mask)
#     print(save_path + u_img_path.split('/')[-1])




# Image_Dir = '/home/venom/projects/data/EPIC-Kitchen100/P04/rgb_frames/P04_107'
# fgbg = cv2.createBackgroundSubtractorMOG2()    #背景分割
# # print(list(glob.glob(Image_Dir + '/*.jpg')))
# for img_path in sorted(list(glob.glob(Image_Dir + '/*.jpg'))):
#     print(img_path)
#     frame = cv2.imread(img_path)
#     fgmask = fgbg.apply(frame)    #应用
    
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     time.sleep(1.5)
#     if k == 27:
#         break

# # cap.release()
# cv2.destroyAllWindows()
img = Image.open('/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/u/frame_0000001333.jpg').convert('P')
print(np.array(img))