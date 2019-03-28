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
pip install --upgrade pip
pip install -r requirements.txt

```

## Creating a synthetic dataset

1. `bin/create_synthetic_dataset.py` creates a synthetic dataset composed by 255x255 images with a centered square. Random colors and noise types are applied. The dataset is divided in subfolders, simulating different videos.

```
python3 bin/create_synthetic_dataset.py --output-dir /path/to/save/dataset/ --num-images 20000 --num-videos 200
```
Parameters to select:
- output-dir: path to save dataset
- num-images: total samples of dataset.
- num-videos: number of dataset subfolders.

NOTE: num_images must be divisible by num_videos

#### **SLURM USERS: edit and run `sh create_synthetic_dataset.sh` instead.**
<br></br>
2. `bin/create_lmdb.py` creates a lmdb file for previous dataset. 

```
python3 bin/create_lmdb.py --data-dir /path/to/synthetic/dataset --output-dir /path/to/synthetic/dataset.lmdb --num-threads 12
```
Parameters to select:
- data-dir: path to synthetic dataset
- output_dir: path where lmdb will be created.

NOTE: include .lmdb extension when entering the path.

NOTE2: Dataset and lmdb file should be in the same directory.

#### **SLURM USERS: edit and run `sh create_lmdb.sh` instead.**
<br></br>
## Training the network

Use `siamfc/training.py` to train the network form scratch using the selected dataset. All training-related parameters are defined in `/siamfc/config.py`. The script will create a directory called `training_exp/NAME_OF_DATASET/timestamp/` to save the result of the experiment, including:
- A copy of python and shell files used in the experiment.
- A `models/` directory to save the network state_dict at every epoch
- A `tensorboard/` directory to supervise the training process using TensorboadX

```
python3 siamfc/training.py --datadir /path/to/dataset
```
Parameters to select:
- datadir: path to dataset to train the network with.

#### **SLURM USERS: edit and run `SBATCH train_siamfc.sh` instead. Output will be writted to slurm.out**
<br></br>

## Creating a synthetic test video

Creates a video composed by a number of frames with a random moving black square on white background. It also creates a `groundtruth.txt` file with target position for each frame. The format followed is `xmin,ymin,height,width`. 

```
python3 bin/create_synthetic_video.py --output-dir /path/to/save/video --image-size 255 --bbox-size 40 --num-frames 300 --max-displacement 4
```
Parameters to select:
- output-dir: path to save the video.
- image-size: size of frames.
- bbox-size: size of squares.
- num_frames: total number of video frames.
- max-displacement: max pixels of displacement allowed between consecutive frames.

#### **SLURM USERS: edit and run `sh create_video.sh` instead.**
<br></br>

## Testing the tracker

Use `siamfc/demo_siamfc.py` to test the tracker with a video. All tracking-related parameters are defined in `siamfc/config.py`. The output will be a directory called `tracking_exp/timestamp` with a copy of the video frames with a predicted bounding box and a graph showing the displacement between grountruth and predicted boundig boxes.
```
python3 siamfc/demo_siamfc.py --video-dir /path/to/video/frames --model-dir /path/to/model.pth
```
 Parameters to select:
- video-dir: directory containing video frames to test the tracker.
- model-dir: path to trained model of the network.

NOTE: we include in this repository a model trained on Imagenet under `models/siamfc_50.pth`

#### **SLURM USERS: edit and run `sh run_tracker.sh` instead.**

## References
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S Fully-Convolutional Siamese Networks for Object Tracking In ECCV 2016 workshops
