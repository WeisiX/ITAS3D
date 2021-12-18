# flownet2-pytorch 

Pytorch implementation of [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925). 

Multiple GPU training is supported, and the code provides examples for training or inference on [MPI-Sintel](http://sintel.is.tue.mpg.de/) clean and final datasets. The same commands can be used for training or inference with other datasets. See below for more detail.

Inference using fp16 (half-precision) is also supported.

For more help, type <br />
    
    python main.py --help

## Network architectures
Below are the different flownet neural network architectures that are provided. <br />
A batchnorm version for each network is available.

 - **FlowNet2S**
 - **FlowNet2C**
 - **FlowNet2CS**
 - **FlowNet2CSS**
 - **FlowNet2SD**
 - **FlowNet2**

## Custom layers

`FlowNet2` or `FlowNet2C*` achitectures rely on custom layers `Resample2d` or `Correlation`. <br />
A pytorch implementation of these layers with cuda kernels are available at [./networks](./networks). <br />
Note : Currently, half precision kernels are not available for these layers.

## Data Loaders

Dataloaders for FlyingChairs, FlyingThings, ChairsSDHom and ImagesFromFolder are available in [datasets.py](./datasets.py). <br />

## Loss Functions

L1 and L2 losses with multi-scale support are available in [losses.py](./losses.py). <br />

## Installation 

    # get flownet2-pytorch source
    git clone https://github.com/NVIDIA/flownet2-pytorch.git
    cd flownet2-pytorch

    # install custom layers
    bash install.sh

## Docker image
Libraries and other dependencies for this project include: Ubuntu 16.04, Python 2.7, Pytorch 0.2, CUDNN 6.0, CUDA 8.0

A Dockerfile with the above dependencies is available 
    
    # Build and launch docker image
    bash launch_docker.sh

## Converted Caffe Pre-trained Models
We've included caffe pre-trained models. Should you use these pre-trained weights, please adhere to the [license agreements](https://drive.google.com/file/d/1TVv0BnNFh3rpHZvD-easMb9jYrPE2Eqd/view?usp=sharing). 

* [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]
* [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
* [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
* [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
* [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
* [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
* [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]
    
## Inference
    # Example on MPISintel Clean   
    python main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean \
    --inference_dataset_root /path/to/mpi-sintel/clean/dataset \
    --resume /path/to/checkpoints 
    
## Training and validation

    # Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
    python main.py --batch_size 8 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
    --training_dataset MpiSintelFinal --training_dataset_root /path/to/mpi-sintel/final/dataset  \
    --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset

    # Example on MPISintel Final and Clean, with MultiScale loss on FlowNet2C model 
    python main.py --batch_size 8 --model FlowNet2C --optimizer=Adam --optimizer_lr=1e-4 --loss=MultiScale --loss_norm=L1 \
    --loss_numScales=5 --loss_startScale=4 --optimizer_lr=1e-4 --crop_size 384 512 \
    --training_dataset FlyingChairs --training_dataset_root /path/to/flying-chairs/dataset  \
    --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset
    
## Results on MPI-Sintel
[![Predicted flows on MPI-Sintel](./image.png)](https://www.youtube.com/watch?v=HtBmabY8aeU "Predicted flows on MPI-Sintel")

## Reference 
If you find this implementation useful in your work, please acknowledge it appropriately and cite the paper using:
````
@InProceedings{IMKDB17,
  author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jul",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
````

## Acknowledgments
Parts of this code were derived, as noted in the code, from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch).
