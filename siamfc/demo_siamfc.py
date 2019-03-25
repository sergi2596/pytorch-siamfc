import os, glob, shutil
import pandas as pd
import argparse
import numpy as np
import torch
import cv2
import time
import sys
from datetime import datetime

sys.path.append(os.getcwd())

from fire import Fire
from tqdm import tqdm

from tracker import SiamFCTracker
from config import config


def main(video_dir, model_dir):

    __FORMATS = ['JPEG', 'jpeg', 'JPG', 'jpg', 'png', 'PNG']
    cwd = os.getcwd()
    siamfc = os.path.join(cwd, 'siamfc')
    tracking_exp = os.path.join(cwd, 'tracking_exp')
    time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    video_name = video_dir.split('/')[-1]
    video_exp_dir = os.path.join(tracking_exp, video_name)
    new_exp_dir = os.path.join(video_exp_dir,time)

    if not os.path.exists(new_exp_dir):
        os.makedirs(new_exp_dir, exist_ok=True)
    
    conf = os.path.join(siamfc,'config.py')
    shell = os.path.join(cwd,'run_tracker.sh')
    shutil.copyfile(conf, new_exp_dir+'/config.py')
    shutil.copyfile(shell, new_exp_dir+'/run_tracker.sh')

    filenames = []
    for item in __FORMATS:
        filenames += glob.glob(os.path.join(video_dir, "*."+item))
    
    filenames = sorted(filenames,
           key=lambda x: int(os.path.basename(x).split('.')[0]))

    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    gt_bboxes = pd.read_csv(os.path.join(video_dir, 'groundtruth.txt'), sep='\t|,| ',
            header=None, names=['xmin', 'ymin', 'width', 'height'],
            engine='python')
    title = video_dir.split('/')[-1]
    device = torch.cuda.current_device()
    
    # starting tracking
    tracker = SiamFCTracker(model_dir, device)
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        else: 
            bbox = tracker.update(frame)
        
        # bbox xmin ymin xmax ymax
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),
                              2)

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        
        cv2.imwrite(new_exp_dir+'/'+str(idx)+'.jpg', frame)

if __name__ == "__main__":
    Fire(main)
