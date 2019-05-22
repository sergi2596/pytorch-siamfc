import os, glob, shutil
import pandas as pd
import argparse
import numpy as np
import torch
import cv2
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from fire import Fire
from tqdm import tqdm
from tracker import SiamFCTracker
from config import config
from scipy.spatial import distance

sys.path.append(os.getcwd())


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
        print('Creating experiment folder...')
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
    print('Loading', str(model_dir)+'...')
    tracker = SiamFCTracker(model_dir, device)
    pred_bboxes = {'xmin':[], 'ymin':[]}
    pred_center = []
    dist = []

    print('Starting tracking...')
    start_time = datetime.now()
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = gt_bboxes.iloc[0].values
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        else: 
            bbox = tracker.update(frame)
        
        pred_bboxes['xmin'].append(int(bbox[0]))
        pred_bboxes['ymin'].append(int(bbox[1]))

        pred_center = ((bbox[0]+((bbox[2] - bbox[0])/2)), (bbox[1]+(bbox[3] - bbox[1])/2))
        gt_center = gt_bboxes.iloc[idx].values
        gt_center = ((int(gt_center[0]+(gt_center[2]/2)-1)), (int(gt_center[1]+(gt_center[3]/2)-1)))
        # print(gt_center, pred_center)
        dist.append(distance.euclidean(gt_center, pred_center))

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
    
    end_time = datetime.now()
    fps = len(frames)/max(1.0, (end_time-start_time).seconds)
    prediction = pd.DataFrame(data=pred_bboxes)
    gt_bboxes.drop('width', axis=1, inplace=True)
    gt_bboxes.drop('height', axis=1, inplace=True)

    displacement = prediction - gt_bboxes
    mean_euclid_dist = np.mean(dist)
    print(mean_euclid_dist)

    x_disp = displacement['xmin'].tolist()
    y_disp = displacement['ymin'].tolist()

    bins = np.linspace(-15, 15, 30)
    plt.subplot(1, 2, 1)
    plt.hist(x_disp, bins, alpha=1)
    plt.title('X displacement (pixels)')
    plt.ylabel('Number of frames')
    plt.subplot(1, 2, 2)
    plt.hist(y_disp, bins, alpha=1, color='g')
    plt.title('Y displacement (pixels)')
    plt.ylabel('Number of frames')
    plt.savefig(new_exp_dir+'/displacement.png')
    print('X mean displacement: {}'.format(np.mean(np.absolute(x_disp))))
    print('Y mean displacement: {}'.format(np.mean(np.absolute(y_disp))))
    print('Euclidean distance: {}'.format(mean_euclid_dist))
    with open(new_exp_dir+'/mean_displacement.txt', 'w+') as file:
        file.write('X mean displacement: {}\n'.format(np.mean(np.absolute(x_disp))))
        file.write('Y mean displacement: {}\n'.format(np.mean(np.absolute(y_disp))))
        file.write('Euclidean distance: {}'.format(mean_euclid_dist))

        file.write('FPS: {}\n'.format(fps))
    print('------ DONE -------')
    print('Results saved to', new_exp_dir)

if __name__ == "__main__":
    Fire(main)
