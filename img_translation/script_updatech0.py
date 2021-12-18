import numpy as np
import os
import glob
from PIL import Image
import argparse


############ INPUT PARAMETERS ############

parser = argparse.ArgumentParser()
parser.add_argument("--group_name", type=str, help="the name of dataset folder")
parser.add_argument("--img_model_name", type=str, help="the model nam for image-to-image translation")
args = parser.parse_args()

group_name = args.group_name
img_model_name = args.img_model_name

############ PROCESS ############

# group_name = 's001_o' #change

seq_folder_list = glob.glob('/ITAS3D/seq_translation/datasets/'+group_name+'/test/ch0/*') 

# input the IMG_MODEL_NAME
p2p_ch0_folder = '/ITAS3D/img_translation/results/'+group_name+'/'+img_model_name+'/test_latest/images/'

for seq_dir in seq_folder_list:
    # print('-----', seq_dir)
    frame_list = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    # print(len(frame_list))
    
#     print(frame_list[:10])
    for v2vframe in frame_list[:5]: #only substitute first 5 frames for each seq
        frame_name = os.path.basename(v2vframe)[:-4]
    #         print(frame_name)
#         print('v2vframe', v2vframe)
        p2pframe_name = frame_name + '_fake_B.png'
        p2pframe_dir = os.path.join(p2p_ch0_folder, p2pframe_name)
#         print('p2pframe_dir', p2pframe_dir)

        #delete v2vframe
        os.remove(v2vframe)

        #copy p2pframe in jpg
        p2p_img = Image.open(p2pframe_dir)
        p2p_img.save(v2vframe)
