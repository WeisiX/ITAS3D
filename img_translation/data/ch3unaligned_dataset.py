import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import random


class Ch3unalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
#         self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

#         self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
#         self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

#         self.A_size = len(self.A_paths)  # get the size of dataset A
#         self.B_size = len(self.B_paths)  # get the size of dataset B

#         btoA = self.opt.direction == 'BtoA'
#         input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
#         output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

#         self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
#         self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        ##
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + '/ch1')  # A1-nuclei
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + '/ch2')  # A2 - cyto
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '/ch0')  # B - CK8

        self.A1_paths = sorted(make_dataset(self.dir_A1, opt.max_dataset_size))   # load images
        self.A2_paths = sorted(make_dataset(self.dir_A2, opt.max_dataset_size))   
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    
        
        self.A1_size = len(self.A1_paths)  # get the size of dataset A1
        self.A2_size = len(self.A2_paths)  # get the size of dataset A2
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output imag
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # assume A1 and A2 have same order and total number
        A1_path = self.A1_paths[index % self.A1_size]  # make sure index is within then range
        A2_path = self.A2_paths[index % self.A2_size]  # make sure index is within then range
        
#         if self.opt.serial_batches:   
#             index_B = index % self.B_size # make sure index is within then range
#         else:   # randomize the index for domain B to avoid fixed pairs.
#             index_B = random.randint(0, self.B_size - 1)
        index_B = random.randint(0, self.B_size - 1) # randomize the index for domain B to avoid fixed pairs.
        B_path = self.B_paths[index_B]
        
#         A_img = Image.open(A_path).convert('RGB')
#         B_img = Image.open(B_path).convert('RGB')

        A1 = Image.open(A1_path).convert('L')
        A2 = Image.open(A2_path).convert('L')
        A3 = A2.filter(ImageFilter.FIND_EDGES)   #the third channel is the edge of cyto   
        A_img = Image.merge("RGB",(A1,A2,A3))
        
        B_img = Image.open(B_path).convert('L')
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A1_size, self.B_size)
