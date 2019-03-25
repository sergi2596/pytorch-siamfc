#!/bin/bash
#SBATCH -J training-test
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080multi
#SBATCH --qos gpu_gtx1080multi
#SBATCH --gres gpu:4

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
# python3 dev/test.py --datadir /home/lv71186/deutsch/pysiamfc/dev/SQUARE_DATASET
# python3 dev/test.py --datadir /home/lv71186/deutsch/datasets/ILSVRC_VID_CURATION
python3 siamfc/training.py --datadir ../datasets/TEST_DATASET