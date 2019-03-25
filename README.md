# pytorch-siamfc

Implementation of SiameseFC for Visual Tracking.

**REPOSITORY UNDER DEVELOPMENT**

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

1. `bin/create_synthetic_dataset.py` creates a synthetic dataset composed by 255x255 images with a centered square. Random colors and noise types are applied. The dataset is divided in subfolders, simulating different videos.
- output-dir: path to save dataset
- num-images: total samples of dataset.
- num-videos: number of dataset subfolders.

**NOTE: num_images must be divisible by num_videos**

```
python3 bin/create_synthetic_dataset.py --output-dir /path/to/save/dataset/ --num-images 20000 --num-videos 200
```
#### **SLURM USERS: edit and run `create_synthetic_dataset.sh` instead**

<br></br>

2. `bin/create_lmdb.py` creates a lmdb file for previous dataset. 

**NOTE: Dataset and lmdb file should be in the same directory**. 

```
python3 bin/create_lmdb.py --data-dir /path/to/synthetic/dataset --output-dir /path/to/synthetic/dataset.lmdb --num-threads 12
```
#### **SLURM USERS: edit and run `create_lmdb.sh` instead**

## Training the network

Use `siamfc/training.py` to train the network form scratch using the selected dataset. All training parameters are defined in `/siamfc/config.py`. The script will create a directory called `training_exp/NAME_OF_DATASET/timestamp/` to save the result of the experiment, including:
- A copy of python and shell files used in the experiment.
- A `models/` directory to save the network state_dict at every epoch
- A `tensorboard/` directory to supervise the training process using TensorboadX

```
python3 siamfc/training.py --datadir /path/to/dataset
```

**NOTE: when working with SLURM, just execute `SBATCH train_siamfc.sh`**


## References
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S Fully-Convolutional Siamese Networks for Object Tracking In ECCV 2016 workshops
