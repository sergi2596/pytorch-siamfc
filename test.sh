#!/bin/bash
#SBATCH -J training-test
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080single
#SBATCH --qos gpu_gtx1080single
#SBATCH --gres gpu:1

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
# python3 test/test.py --datadir /home/lv71186/deutsch/pysiamfc/test/SQUARE_DATASET
python3 test/test.py --datadir /home/lv71186/deutsch/datasets/ILSVRC_VID_CURATION