#!/bin/bash
#SBATCH -J training-test
#SBATCH -N 2
#SBATCH --partition gpu_gtx1080multi
#SBATCH --qos gpu_gtx1080multi
#SBATCH --gres gpu:2

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 siamfc/main.py --datadir /home/lv71186/deutsch/datasets/ILSVRC_VID_CURATION
