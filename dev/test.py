import argparse
import os, glob, shutil
from datetime import datetime
import time

import sys

import pickle
import lmdb
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR


from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from config import config
import siamesenet as siamnet
from datasets import ImagnetVIDDataset
from custom_transforms import Normalize, ToTensor, RandomStretch, \
    RandomCrop, CenterCrop, RandomBlur, ColorAug


# from bokeh.plotting import figure
# from bokeh.io import show
# from bokeh.models import LinearAxis, Range1d
# import numpy as np


parser = argparse.ArgumentParser(description='PyTorch SiamFC Training')
parser.add_argument('--datadir', metavar='DIR', help='path to dataset')

best_prec1 = 0


def main():

    # create Experiment directories
    cwd = os.getcwd()
    test = os.path.join(cwd,'test')
    experiment_folder = os.path.join(test, config.experiment_folder)
    time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    print('\n================= EXPERIMENT START TIME',
          time, '=================\n')
    
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    new_exp_dir = os.path.join(test, config.experiment_folder, time)
    tensorboard_dir = os.path.join(new_exp_dir+'/tensorboard/')
    models_dir = os.path.join(new_exp_dir+'/models/')
    output_files = []

    if not os.path.exists(new_exp_dir):
        os.mkdir(new_exp_dir)
        os.mkdir(tensorboard_dir)
        os.mkdir(models_dir)
        print('Experiment folder created')

        for file in glob.glob(os.path.join(test, "*.py")):
            if os.path.isfile(file):
                shutil.copyfile(file, new_exp_dir+file.split(test)[1])

        for file in glob.glob(os.path.join(cwd, "*.out")):
            output_files.append(file)
    
    # Create Tensorboard summary writer
    writer = SummaryWriter(tensorboard_dir)

    global args, best_prec1
    args = parser.parse_args()
    loss_list = []
    acc_list = []

    if 'ILSVRC_VID_CURATION' in args.datadir:
        # loading meta data
        meta_data_path = os.path.join(args.datadir, "meta_data.pkl")
        meta_data = pickle.load(open(meta_data_path, 'rb'))
        videos = [x[0] for x in meta_data]

    elif 'SQUARE_DATASET' in args.datadir:
        videos = ['images_1', 'images_2', 'images_3',
                  'images_4', 'images_5', 'images_6']

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(
        videos, test_size=1-config.train_ratio)

    # define transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    random_crop_size = config.instance_size - 2 * config.total_stride
    train_reference_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        Normalize(),
        ToTensor()
    ])
    train_search_transforms = transforms.Compose([
        # RandomCrop((random_crop_size, random_crop_size), config.max_translate),
        Normalize(),
        ToTensor()
    ])
    valid_reference_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        Normalize(),
        ToTensor()
    ])
    valid_search_transforms = transforms.Compose([
        Normalize(),
        ToTensor()
    ])

    # opem lmdb
    db = lmdb.open(args.datadir+'.lmdb', readonly=True, map_size=int(50e9))

    # create dataset
    train_dataset = ImagnetVIDDataset(db, train_videos, args.datadir,
                                      train_reference_transforms, train_search_transforms)
    valid_dataset = ImagnetVIDDataset(db, valid_videos, args.datadir,
                                      valid_reference_transforms, valid_search_transforms, training=False)

    # create dataloader
    print('Loading Train Dataset...')
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    print('Loading Validation Dataset...')
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                             shuffle=False, pin_memory=True, num_workers=config.valid_num_workers, drop_last=True)


    print('Loading SiameseNet')
    model = siamnet.SiameseNet()
    model.features = torch.nn.DataParallel(model.features)
    model.init_weights()
    model = model.cuda()
    print("Available GPUs:", torch.cuda.device_count())
    print("Model running on GPU:", next(model.parameters()).is_cuda), '\n\n'
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    scheduler = StepLR(optimizer, step_size=config.step_size,
                       gamma=config.gamma)

    for epoch in range(config.start_epoch, config.epoch):

        # model.train() tells your model that you are training the model.
        # So effectively layers like dropout, batchnorm etc. which behave
        # different on the train and test procedures

        training_loss = []
        model.train()

        for i, data in enumerate(tqdm(trainloader)):

            reference_imgs, search_imgs = data

            # Variable is a thin wrapper around a Tensor object,
            # that also holds the gradient w.r.t. to it.
            reference_var = Variable(reference_imgs).cuda()
            search_var = Variable(search_imgs).cuda()

            # we need to set the gradients to zero before starting
            # to do backpropragation because PyTorch accumulates
            # the gradients on subsequent backward passes.
            optimizer.zero_grad()

            outputs = model(reference_var, search_var)
            loss = model.loss(outputs)
            loss.backward()
            optimizer.step()

            step = epoch * len(trainloader) + i
            writer.add_scalars('Loss', {'Training': loss.data}, step)

            training_loss.append(loss.data)

        training_loss = torch.mean(torch.stack(training_loss)).item()
        valid_loss = []
        model.eval()

        for i, data in enumerate(tqdm(validloader)):

            reference_imgs, search_imgs = data
            reference_var = Variable(reference_imgs.cuda())
            search_var = Variable(search_imgs.cuda())
            outputs = model(reference_var, search_var)
            loss = model.loss(outputs)
            valid_loss.append(loss.data)

        valid_loss = torch.mean(torch.stack(valid_loss)).item()

        print("EPOCH %d Training Loss: %.4f, Validation Loss: %.4f" %
              (epoch, training_loss, valid_loss))

        torch.save(model.cpu().state_dict(), models_dir +
                   "siamfc_{}.pth".format(epoch+1))
        writer.add_scalars('Loss', {'Validation':valid_loss}, (epoch+1)*len(trainloader))

        model.cuda()
        scheduler.step()

    time = datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
    print('\n================= EXPERIMENT END TIME', time, '=================\n')

    # Copy slurm output files to experiment folder
    for file in output_files:
        shutil.copyfile(file, new_exp_dir+file.split(cwd)[1])


if __name__ == '__main__':
    main()