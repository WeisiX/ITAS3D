import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset, make_dataset_frame_from_seq
from PIL import Image, ImageFilter


class FrameseqDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
#         self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
#         self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + '/ch1')  # create a path '/path/train/ch-' A1-nuclei
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + '/ch2')  # A2 - cyto
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '/ch0')  # B - CK8
        print('self.dir_B', self.dir_B)

        self.A1_paths = sorted(make_dataset_frame_from_seq(self.dir_A1, opt.max_dataset_size))   # load images from '/path/train/ch-'
        self.A2_paths = sorted(make_dataset_frame_from_seq(self.dir_A2, opt.max_dataset_size))   
        self.B_paths = sorted(make_dataset_frame_from_seq(self.dir_B, opt.max_dataset_size)) 
        
        self.A1_size = len(self.A1_paths)  # get the size of dataset A1
        self.A2_size = len(self.A2_paths)  # get the size of dataset A2
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print('self.B_size', self.B_size)
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A1_path = self.A1_paths[index]
        A2_path = self.A2_paths[index]
        B_path = self.B_paths[index]
        
#         AB = Image.open(AB_path).convert('RGB')
        A1 = Image.open(A1_path).convert('L')
        A2 = Image.open(A2_path).convert('L')
        A3 = A2.filter(ImageFilter.FIND_EDGES)   #the third channel is the edge of cyto   
        A = Image.merge("RGB",(A1,A2,A3))
        
        B = Image.open(B_path).convert('L')
        
#         # split AB image into A and B
#         w, h = AB.size
#         w2 = int(w / 2)
#         A = AB.crop((0, 0, w2, h))
#         B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.B_paths)
