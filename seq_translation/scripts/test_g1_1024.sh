## Step 1: single-level image translation

source activate img_translation

python /ITAS3D/img_translation/test.py --dataroot /ITAS3D/seq_translation/datasets/${DATASET_NAME} --checkpoints_dir /ITAS3D/img_translation/checkpoints --name ${IMG_MODEL_NAME} --model pix2pix --netG unet_512 --direction AtoB --dataset_mode frameseqtest --norm batch --input_nc 3 --output_nc 1 --results_dir /ITAS3D/img_translation/results/${DATASET_NAME} --num_test 100000 --load_size 1024 --crop_size 1024 

python /ITAS3D/img_translation/script_updatech0.py --group_name ${DATASET_NAME} --img_model_name ${IMG_MODEL_NAME}

## Step 2: image-sequence translation
conda deactivate
source activate seq_translation

python /ITAS3D/seq_translation/test.py --name ${SEQ_MODEL_NAME} --dataroot /ITAS3D/seq_translation/datasets/${DATASET_NAME} --checkpoints_dir /ITAS3D/seq_translation/checkpoints --dataset_mode w1_test --output_nc 3 --loadSize 1024 --n_scales_spatial 1 --n_downsample_G 2 --use_real_img --results_dir /ITAS3D/seq_translation/results/${DATASET_NAME} --how_many 100000

conda deactivate