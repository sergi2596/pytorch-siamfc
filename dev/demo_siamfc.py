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


def main(video_dir, gt_dir, model_dir):

    __FORMATS = ['JPEG', 'jpeg', 'JPG', 'jpg', 'png', 'PNG']
    cwd = os.getcwd()
    results = os.path.join(cwd,'dev',config.test_folder)
    time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    experiment = os.path.join(results,time)

    if not os.path.exists(experiment):
        os.mkdir(experiment)
    
    conf = os.path.join(cwd,'dev','config.py')
    shell = os.path.join(cwd,'run_tracker.sh')
    shutil.copyfile(conf, experiment+'/config.py')
    shutil.copyfile(shell, experiment+'/run_tracker.sh')

    filenames = []
    for item in __FORMATS:
        filenames += glob.glob(os.path.join(video_dir, "*."+item))
    
    filenames = sorted(filenames,
           key=lambda x: int(os.path.basename(x).split('.')[0]))

    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    gt_bboxes = pd.read_csv(os.path.join(gt_dir), sep='\t|,| ',
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
        
        cv2.imwrite(experiment+'/'+str(idx)+'.jpg', frame)

if __name__ == "__main__":
    Fire(main)
