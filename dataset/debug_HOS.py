from preprocess_flow import proprocess_flow
from PIL import Image
import numpy as np 

flow_preprocessor = proprocess_flow()
img = '/home/venom/projects/EgoHOS/testimages/images/frame_0000002761.jpg'
pil_img = np.array(Image.open(img))

flow_u = np.array(Image.open('/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/u/frame_0000001325.jpg').convert('P'))
flow_v = np.array(Image.open('/home/venom/projects/XMem/val_data/P01/flow_frames/P01_11/P01_11_9/v/frame_0000001325.jpg').convert('P'))
flow = np.concatenate([[flow_u], [flow_v]]).transpose(1,2,0)

print(flow.shape)
print(pil_img.shape)
after_process = flow_preprocessor.remove_hand_region(pil_img, flow)
# after_process = Image.fromarray(after_process)
positions_zeros = np.where(after_process)
pil_img[positions_zeros] = [0,0,0]

# palette = [0,0,0,255,255,255,128,0,0,0,128,0,0,0,128,255,0,0,255,255,0]
# others = list(np.random.randint(0,255,size=256*3-len(palette)))
# palette.extend(others)

# after_process.putpalette(palette)
# after_process.show()