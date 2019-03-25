# pytorch-siamfc

Implementation of SiameseFC for Visual Tracking.

Repository under development.

**Code adapted from https://github.com/StrangerZhang/SiamFC-PyTorch.git**

## Initial setup

```
git clone https://github.com/sergi2596/pytorch-siamfc.git
cd pytorch-siamfc
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

```

## Creating a synthetic dataset

`create_synthetic_dataset.sh` creates a synthetic dataset composed by 255x255 images with a centered square. Random colors and noise types are applied. The dataset is divided in subfolders, simulating different videos.
- num-images: total samples of dataset.
- num-videos: number of dataset subfolders.

**NOTE**: num_images must be divisible by num_videos

```
#!/bin/bash
#SBATCH -J create-dataset
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080multi
#SBATCH --qos gpu_gtx1080multi
#SBATCH --gres gpu:4

module purge
module load gcc/6.4 python/3.6 cuda/9.1.85
python3 bin/create_synthetic_dataset.py --output-dir /path/to/save/dataset/ --num-images 20000 --num-videos 200
```

## References
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S Fully-Convolutional Siamese Networks for Object Tracking In ECCV 2016 workshops
