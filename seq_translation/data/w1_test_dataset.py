### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from PIL import Image, ImageFilter
import numpy as np

class W1testDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + '/ch1')  # create a path '/path/train/ch-' A1-nuclei
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + '/ch2')  # A2 - cyto
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '/ch0')  # B - CK8
        self.use_real = opt.use_real_img
        self.A_is_label = self.opt.label_nc != 0

#         self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.A1_paths = sorted(make_grouped_dataset(self.dir_A1))   # load images from '/path/train/ch-'
        self.A2_paths = sorted(make_grouped_dataset(self.dir_A2)) 
        if self.use_real:
            self.B_paths = sorted(make_grouped_dataset(self.dir_B))
            check_path_valid(self.A1_paths, self.B_paths)
#         if self.opt.use_instance:                
#             self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
#             self.I_paths = sorted(make_grouped_dataset(self.dir_inst))
#             check_path_valid(self.A_paths, self.I_paths)

        self.init_frame_idx(self.A1_paths) #?
        
        
        ## w1 dataset - training phase
#         self.opt = opt
#         self.root = opt.dataroot
        
#         self.dir_A1 = os.path.join(opt.dataroot, opt.phase + '/ch1')  # create a path '/path/train/ch-' A1-nuclei
#         self.dir_A2 = os.path.join(opt.dataroot, opt.phase + '/ch2')  # A2 - cyto
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + '/ch0')  # B - CK8
#         self.A_is_label = self.opt.label_nc != 0

#         self.A1_paths = sorted(make_grouped_dataset(self.dir_A1))   # load images from '/path/train/ch-'
#         self.A2_paths = sorted(make_grouped_dataset(self.dir_A2))   
#         self.B_paths = sorted(make_grouped_dataset(self.dir_B))
#         check_path_valid(self.A1_paths, self.B_paths)
#         check_path_valid(self.A2_paths, self.B_paths)
        
#         self.n_of_seqs = len(self.A1_paths)                 # number of sequences to train       
#         self.seq_len_max = max([len(A1) for A1 in self.A1_paths])        
#         self.n_frames_total = self.opt.n_frames_total      # current number of frames to train in a single iteration

    def __getitem__(self, index):
        self.A1, self.B, self.I, seq_idx = self.update_frame_idx(self.A1_paths, index)
        tG = self.opt.n_frames_G
        
        #this part should be good
        A1_img = Image.open(self.A1_paths[seq_idx][0]).convert('L')        
        params = get_img_params(self.opt, A1_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB
        
        frame_range = list(range(tG)) if self.A1 is None else [tG-1]
           
        self.I = 0
        for i in frame_range:                                                   
            A1_path = self.A1_paths[seq_idx][self.frame_idx + i]
            A2_path = self.A2_paths[seq_idx][self.frame_idx + i]
            Ai = self.get_image_2ch(A1_path, A2_path, transform_scaleA)            
            self.A = concat_frame(self.A, Ai, tG)

            if self.use_real:
                B_path = self.B_paths[seq_idx][self.frame_idx + i]
                Bi = self.get_image(B_path, transform_scaleB, is_single=False)                
                self.B = concat_frame(self.B, Bi, tG)
            else:
                self.B = 0

#             if self.opt.use_instance:
#                 I_path = self.I_paths[seq_idx][self.frame_idx + i]
#                 Ii = self.get_image(I_path, transform_scaleA) * 255.0                
#                 self.I = concat_frame(self.I, Ii, tG)
#             else:
#                 self.I = 0

        self.frame_idx += 1        
        return_list = {'A': self.A, 'B': self.B, 'inst': self.I, 'A_path': A1_path, 'change_seq': self.change_seq}
        return return_list
    
    
        ##w1dataset
#         tG = self.opt.n_frames_G
        
#         A1_paths = self.A1_paths[index % self.n_of_seqs]
#         A2_paths = self.A2_paths[index % self.n_of_seqs]
#         B_paths = self.B_paths[index % self.n_of_seqs]                              
        
#         # setting parameters
#         n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A1_paths), index)  

#         # setting transformers
#         B_img = Image.open(B_paths[start_idx]).convert('L') #ck8 only 1 channel, do this to get size
     
#         params = get_img_params(self.opt, B_img.size)   #size:(width,height)       
#         transform_scaleB = get_transform(self.opt, params)
#         transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB # remove A_is_label in the future

#         # read in images
#         A = B = inst = 0
#         for i in range(n_frames_total):            

#             A1_path = A1_paths[start_idx + i * t_step]
#             A2_path = A2_paths[start_idx + i * t_step]
#             B_path = B_paths[start_idx + i * t_step]            

#             Ai = self.get_image_2ch(A1_path, A2_path, transform_scaleA)
            
#             Bi = self.get_image(B_path, transform_scaleB, is_single=False)
            
#             A = Ai if i == 0 else torch.cat([A, Ai], dim=0)            
#             B = Bi if i == 0 else torch.cat([B, Bi], dim=0)                           

#         return_list = {'A': A, 'B': B, 'inst': inst, 'A_path': A1_path, 'B_paths': B_path}
#         return return_list


#     def get_image(self, A_path, transform_scaleA, is_label=False):
#         A_img = Image.open(A_path)
#         A_scaled = transform_scaleA(A_img)
#         if is_label:
#             A_scaled *= 255.0
#         return A_scaled
    
    def get_image(self, A_path, transform_scaleA, is_label=False, is_single=False):
        if is_single:
            A_img = Image.open(A_path).convert('L')
        else:
            A_img = Image.open(A_path).convert('RGB')        
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled
    
    def get_image_2ch(self, A1_path, A2_path, transform_scaleA):
        A1_img = Image.open(A1_path).convert('L')
        A2_img = Image.open(A2_path).convert('L')
        A3_img = A2_img.filter(ImageFilter.FIND_EDGES)   #the third channel is the edge of cyto   
        A_img = Image.merge("RGB",(A1_img,A2_img,A3_img))
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def __len__(self):        
        return sum(self.frames_count)

    def n_of_seqs(self):        
        return len(self.A1_paths)

    def name(self):
        return 'W1testDataset'