#!/bin/bash
#SBATCH -J training-test
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080single
#SBATCH --qos gpu_gtx1080single
#SBATCH --gres gpu:1

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 dev/create_synthetic_video.py --output-dir ../datasets/synthetic_video/ --image-size 255 --bbox-size 40 --num-frames 300 --max-displacement 4