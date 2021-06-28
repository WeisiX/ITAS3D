# ITAS3D

Pytorch implementation for the image translation assisted segmentation in 3D (ITAS3D), an annotation-free 3D gland-segmentation method based on generative image-sequence translation, which allowed us to extract histomorphometric glandular features. 

This pipeline consists of two steps: the image-sequence translation from the fluorescent analog of H&E histology images to CK8 immunofluorescence (initiated with single-level image translation), and the 3D segmentation of glands based on the synthetic CK8.

<img src="https://github.com/WeisiX/ITAS3D/blob/master/img/overview.png" width="600px"/>

The code and user instrctions borrow heavily from [Video-to-Video Synthesis](https://tcwang0509.github.io/vid2vid/) and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Image-sequence translation training

### Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 0.4

### Installation
- Clone this repo:
```bash
git clone https://github.com/WeisiX/ITAS3D
cd ITAS3D/seq_translation
```
- The dependencies are available in `ITAS3D/img_translation/environment.yml`
- We strongly suggest to create an individual conda environment for the image-sequence translation, for example, `seq_translation`.

### Training
- First, download the FlowNet2 checkpoint file by running `python scripts/download_models_flownet2.py`.

- Training with a single GPU:
  - We trained our models with a 12-GB GPU (NVIDIA Tesla P100) at the targeted resolution (1024 x 1024 for each level in the image sequence). 
  - For example, we provided a sample training script (`/ITAS3D/seq_translation/scripts/train_g1_1024.sh`)
 
  ```bash
  source activate seq_translation
  python /ITAS3D/seq_translation/train.py --name [seq_model_name] --dataroot /ITAS3D/seq_translation/datasets/[dataset_name]/ --checkpoints_dir /ITAS3D/seq_translation/checkpoints --dataset_mode w1 --output_nc 3 --loadSize 800 --n_downsample_G 2 --n_frames_D 2 --num_D 3 --max_frames_per_gpu 1 --n_frames_total 4 --niter_step 2 
  ```
  
## Single-level image translation training

### Prerequisites
- Linux or OSX
- Python 3
- NVIDIA GPU + CUDA CuDNN

### Installation
- Clone this repo:
```bash
git clone https://https://github.com/WeisiX/ITAS3D
cd ITAS3D/img_translation
```
- The dependencies are available in the ITAS3D/img_translation/environment.yml
- We strongly suggest to create an individual conda environment for the single-level image translation, for example, `img_translation`.

### Training

- Training with a single GPU:
  - We trained our models with a 12-GB GPU (NVIDIA Tesla P100) at the targeted resolution (1024 x 1024 for each level in the image sequence). 
  - For example, we provided a sample training script (`/ITAS3D/img_translation/scripts/train_xxxx.sh`)
 
  ```bash
  source activate img_translation
  python /ITAS3D/img_translation/train.py --dataroot /ITAS3D/img_translation/datasets/[dataset_name] --checkpoints_dir /ITAS3D/img_translation/checkpoints --name [img_model_name] --model pix2pix --netG unet_512 --direction AtoB --lambda_L1 100 --dataset_mode frameseq --norm batch --pool_size 0 --input_nc 3 --output_nc 1 --load_size 1024 --crop_size 512 --display_id 0
  ```

## Testing/Inference of image-sequence translation

- Sample test case can be downloaded with 
```bash
cd /ITAS3D/seq_translation/
python ./scripts/download_datasets_ITAS3D.py
```
- Trained models can be downloaded with 
```bash
cd /ITAS3D/seq_translation/
python ./scripts/download_models_ITAS3D.py
```

- We provided a sample test script(`/ITAS3D/seq_translation/scripts/test_g1_1024.sh`)

```bash
## Step 1: single-level image translation

source activate img_translation

python /ITAS3D/img_translation/test.py --dataroot /ITAS3D/seq_translation/datasets/[dataset_name] --checkpoints_dir /ITAS3D/img_translation/checkpoints --name [img_model_name] --model pix2pix --netG unet_512 --direction AtoB --dataset_mode frameseqtest --norm batch --input_nc 3 --output_nc 1 --results_dir /ITAS3D/img_translation/results/[dataset_name] --num_test 100000 --load_size 1024 --crop_size 1024 

python /ITAS3D/img_translation/sciprt_updatech0.py --group_name [dataset_name]

## Step 2: image-sequence translation
conda deactivate
source activate seq_translation

python /ITAS3D/seq_translation/test.py --name [seq_model_name] --dataroot /ITAS3D/seq_translation/datasets/[dataset_name] --checkpoints_dir /ITAS3D/seq_translation/checkpoints --dataset_mode w1_test --output_nc 3 --loadSize 1024 --n_scales_spatial 1 --n_downsample_G 2 --use_real_img --results_dir /ITAS3D/seq_translation/results/[dataset_name] --how_many 100000

conda deactivate
```

## Other details
- For more training/test tips and details, please refer to [Video-to-Video Synthesis](https://tcwang0509.github.io/vid2vid/) and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## 3D gland segmentation based on synthetic CK8 immunofluorescence images
- Please see code in `/ITAS3D/segmentation.ipynb`
