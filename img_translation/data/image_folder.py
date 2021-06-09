"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
    
    
def make_dataset_frame_from_seq(dir, max_dataset_size=float("inf"),total_n_frame_per_seq = 50, select_n_frame_per_seq = 4):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    ####
    seq_list = glob.glob(os.path.join(dir, '*'))
    print('os.path.join(dir, s*)=', os.path.join(dir, '*'))
    print('len(seq_list)=', len(seq_list))
    for seq_folder in seq_list:
        frame_list = glob.glob(os.path.join(seq_folder, '*.jpg'))
        for idx in range(select_n_frame_per_seq):
            if select_n_frame_per_seq==total_n_frame_per_seq: #use all images in the seq
                frame_num = idx
            else:
                frame_num = int((total_n_frame_per_seq//(select_n_frame_per_seq+1))*(idx+1))
            images.append(frame_list[frame_num]) 
    
    ####
    return images[:min(max_dataset_size, len(images))]
    
    
###add new for test
def make_dataset_frame_from_seq_test(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    ####
    seq_list = glob.glob(os.path.join(dir, '*'))
    print('os.path.join(dir, s*)=', os.path.join(dir, '*'))
    print('len(seq_list)=', len(seq_list))
    for seq_folder in seq_list:
        frame_list = sorted(glob.glob(os.path.join(seq_folder, '*.jpg')))
        for idx in range(5) : #range(len(frame_list)) #now only process the first 5 images of every block for the purpose of initiating seq generation
            images.append(frame_list[idx]) 
    ####
    return images[:min(max_dataset_size, len(images))]
###


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
