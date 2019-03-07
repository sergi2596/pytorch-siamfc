#!/bin/bash
#SBATCH -J training-test
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080multi
#SBATCH --qos gpu_gtx1080multi
#SBATCH --gres gpu:4

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 siamfc/main.py --datadir /path/to/ILSVRC_VID_CURATION
