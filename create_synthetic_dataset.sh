#!/bin/bash
#SBATCH -J create-dataset
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080multi
#SBATCH --qos gpu_gtx1080multi
#SBATCH --gres gpu:4

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 bin/create_synthetic_dataset.py --output-dir /path/to/save/dataset --num-images 20000 --num-videos 200
