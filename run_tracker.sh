#!/bin/bash
#SBATCH -J run-tracker
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080single
#SBATCH --qos gpu_gtx1080single
#SBATCH --gres gpu:1

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 siamfc/demo_siamfc.py --video-dir /path/to/video --model-dir /path/to/model.pth
