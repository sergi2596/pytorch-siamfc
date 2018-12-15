#!/bin/bash
#SBATCH -J training-test
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080multi
#SBATCH --qos gpu_gtx1080multi
#SBATCH --gres gpu:1
#SBATCH --mail-type=ALL    # first have to state the type of event to occur  (BEGIN, END, FAIL, REQUEUE, ALL)
#SBATCH --mail-user=sergi.sanchez.deutsch@gmail.com  # and then your email address

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python siamfc/main.py -a siamesenet --lr 0.01 /home/lv71186/deutsch/datasets/ILSVRC/Data/VID
