#!/bin/bash
#SBATCH -J run-tracker
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080single
#SBATCH --qos gpu_gtx1080single
#SBATCH --gres gpu:1

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
# python3 siamfc/demo_siamfc.py --video-dir /home/lv71186/deutsch/datasets/lasot/boat-13 --model-dir /home/lv71186/deutsch/pytorch-siamfc/training_exp/SYNTHETIC_DATASET/11-04-2019_19-27-45/models/siamfc_7.pth
# python3 siamfc/demo_siamfc.py --video-dir /home/lv71186/deutsch/datasets/lasot/boat-13 --model-dir --model-dir /home/lv71186/deutsch/pytorch-siamfc/training_exp/ILSVRC_VID_CURATION/07-05-2019_19-22-36/models/siamfc_50.pth
python3 siamfc/demo_siamfc.py --video-dir /home/lv71186/deutsch/datasets/video_16 --model-dir --model-dir /home/lv71186/deutsch/pytorch-siamfc/training_exp/ILSVRC_VID_CURATION/07-05-2019_19-22-36/models/siamfc_50.pth
# python3 siamfc/demo_siamfc.py --video-dir /home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00044000 --model-dir /home/lv71186/deutsch/pytorch-siamfc/training_exp/SYNTHETIC_DATASET/11-04-2019_19-27-45/models/siamfc_7.pth
# python3 siamfc/demo_siamfc.py --video-dir /home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00160000  --model-dir /home/lv71186/deutsch/pytorch-siamfc/training_exp/ILSVRC_VID_CURATION/07-05-2019_19-22-36/models/siamfc_50.pth

#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000
#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00007009
#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00016000