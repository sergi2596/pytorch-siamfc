#!/bin/bash
#SBATCH -J training
#SBATCH -N 1
#SBATCH --partition gpu_k20m
#SBATCH --qos gpu_k20m
#SBATCH --gres gpu:2

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 siamfc/training.py --loss bce --datadir /home/lv71186/deutsch/datasets/SYNTHETIC_DATASET
# python3 siamfc/training.py --loss bce --datadir /home/lv71186/deutsch/datasets/ILSVRC_VID_CURATION
