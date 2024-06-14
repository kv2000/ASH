#!/bin/bash
#SBATCH --job-name=lets_learn_to_meow
#SBATCH -a 0-114%20

python get_image.py \
    --camera_id $SLURM_ARRAY_TASK_ID \
    --input_video_dir './raw_data/Subject0022/training/videos/' \
    --input_mask_dir './raw_data/Subject0022/training/foregroundSegmentation/' \
    --output_img_dir './raw_data/Subject0022/metadata/training/images/' \
    --output_mask_dir './raw_data/Subject0022/metadata/training/foregroundSegmentation/'