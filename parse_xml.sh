#!/bin/bash
#SBATCH -J parse_xml
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080multi
#SBATCH --qos gpu_gtx1080multi
#SBATCH --gres gpu:1

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 bin/xml_parser.py --annotations-dir /home/lv71186/deutsch/datasets/ILSVRC2015/Annotations/VID/train/val/ILSVRC2015_val_00160000

#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000
#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00007009
#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00016000
#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00044000
# ILSVRC2015_val_00159002