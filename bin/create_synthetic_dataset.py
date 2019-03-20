from tqdm import tqdm
from fire import Fire
from random import randint
import os
import sys
import glob
import random
import numpy as np
import cv2
import json
import functools
from multiprocessing import Pool
from tqdm import tqdm


sys.path.append(os.getcwd())

__IMAGE_SIZE = 255
__TARGET_SIZE = 60
__NOISE_TYPES = ['gauss', 's&p', 'speckle']


def random_color():
    '''Creates tuple with random RGB values
    
    Returns:
        [tuple] -- [Random RGB values]
    '''

    return (randint(0, 255), randint(0, 255), randint(0, 255))


def noisy(noise_typ, image):
    '''Adds random noise to image

    Arguments:
        noise_typ {string} -- [Type of noise]
        image {numpy.ndarray} -- [image]

    Returns:
        [numpy.ndarray] -- [Image with noise]
    '''

    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        # var = 0.1
        # sigma = var**0.5
        gauss = np.random.normal(mean, 1, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    # elif noise_typ == "poisson":
    #     vals = len(np.unique(image))
    #     vals = 2 ** np.ceil(np.log2(vals))
    #     noisy = np.random.poisson(image * vals) / float(vals)
    #     return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def worker(output_dir, img_x_video, video_dir):
    '''Creates a synthetic image with random background and target color
    and (img_x_video) different copies with random noise.
    
    Arguments:
        output_dir {string} -- [Path to dataset]
        img_x_video {int} -- [Number of images of subfolder]
        video_dir {string} -- [Path to subfolder]
    
    Returns:
        [string, dict] -- [Metadata of video]
    '''


    background = np.zeros((__IMAGE_SIZE, __IMAGE_SIZE, 3))
    background[:] = random_color()
    # At the moment, target is always a square. ymin would be equal to xmin
    xmin = (int(__IMAGE_SIZE/2)-int(__TARGET_SIZE/2))
    xmax = (int(__IMAGE_SIZE/2)+int(__TARGET_SIZE/2))
    upper_left = (xmin, xmin)
    bottom_right = (xmax, xmax)
    template = cv2.rectangle(background, upper_left, bottom_right, random_color(), -1)
    video_name = video_dir.split('/')[-1]
    filenames = {0: []}

    for i in range(img_x_video):
        image = template
        image = noisy(__NOISE_TYPES[randint(0, len(__NOISE_TYPES)-1)], image)
        cv2.imwrite(video_dir+'/'+str(i)+'.00.x.jpg', image)
        filenames[0].append(str(i))

    return video_name, filenames


def create_dataset(output_dir, num_images, num_videos, num_threads=32):
    '''Creates a dataset of synthetic images consistent in a centered 
    square with colored background. Images are separated into different
    subfolders, simulating different videos. 

    Arguments:
        output_dir {string} -- [Directory to save dataset]
        num_images {int} -- [Total images of dataset]
        num_videos {int} -- [Number of videos (or subfolders)]

    Keyword Arguments:
        num_threads {int} -- [Number of threads] (default: {32})
    '''

    data_dir = os.path.join(output_dir)
    assert num_images >= num_videos, "num_subfolders cannot be greater than num_images"
    assert not os.path.exists(
        data_dir), "output_dir already exists. Please, choose a different name"
    assert num_images % num_videos == 0, "num_images/num_videos should be an integer. Choose other values"

    img_x_video = int(num_images/num_videos)
    os.mkdir(data_dir)
    all_videos = []
    for i in range(num_videos):
        name = 'video_'+str(i)
        subfolder = os.path.join(data_dir, name)
        os.mkdir(subfolder)
        all_videos.append(subfolder)

    metadata = {}
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(
                functools.partial(worker, output_dir, img_x_video), all_videos), total=len(all_videos)):

            metadata[ret[0]] = ret[1]

    output_file = os.path.join(data_dir, 'metadata.txt')
    with open(output_file, 'w') as file:
        json.dump(metadata, file)


if __name__ == '__main__':
    Fire(create_dataset)
