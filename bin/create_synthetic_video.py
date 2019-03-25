import os, random
import numpy as np
import cv2
import sys

sys.path.append(os.getcwd())

from fire import Fire
from tqdm import tqdm


def main(output_dir, image_size, bbox_size, num_frames, max_displacement):

    cwd = os.getcwd()
    video_dir = os.path.join(output_dir)
    gt = os.path.join(video_dir,'groundtruth.txt')
    if not os.path.exists(video_dir):
        os.mkdir(video_dir) 

    with open(gt, 'w+') as file:
        for i in tqdm(range(num_frames)):
            white_frame = np.zeros((image_size, image_size, 3))
            white_frame[:] = (255,255,255)

            if i == 0:
                xmin = random.randint(0, image_size-bbox_size)
                ymin = random.randint(0, image_size-bbox_size)
                pos = [xmin, ymin, xmin+bbox_size, ymin+bbox_size]     
            else:
                x_disp = random.randint(-max_displacement, max_displacement)
                y_disp = random.randint(-max_displacement, max_displacement)
                new_pos = [pos[0]+x_disp, pos[1]+y_disp, pos[2]+x_disp, pos[3]+y_disp]
                if new_pos[0] >= 0 and new_pos[2] < image_size-1:
                    pos[0] = new_pos[0]
                    pos[2] = new_pos[2]
                if new_pos[1] >= 0 and new_pos[3] < image_size-1:
                    pos[1] = new_pos[1]
                    pos[3] = new_pos[3]

            frame = cv2.rectangle(white_frame, (pos[0], pos[1]), (pos[2], pos[3]), (0,0,0), -1)
            cv2.imwrite(video_dir+'/'+str(i)+'.jpg', frame)
            file.write(str(pos[0])+','+str(pos[1])+','+str(bbox_size)+','+str(bbox_size)+'\n')

if __name__ == "__main__":
    Fire(main)