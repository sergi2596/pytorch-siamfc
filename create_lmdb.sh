#!/bin/bash
#BATCH -J training-test
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080single
#SBATCH --qos gpu_gtx1080single
#SBATCH --gres gpu:1

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 bin/create_lmdb.py --data-dir /path/to/synthetic/dataset --output-dir /path/to/synthetic/dataset.lmdb --num-threads 12

