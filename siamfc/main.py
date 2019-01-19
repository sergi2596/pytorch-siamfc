import argparse
import os
import shutil
import time

import sys

import pickle
import lmdb
from sklearn.model_selection import train_test_split
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


# from bokeh.plotting import figure
# from bokeh.io import show
# from bokeh.models import LinearAxis, Range1d
# import numpy as np


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--datadir', metavar='DIR', help='path to dataset')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_prec1 = 0


def main():

    global args, best_prec1
    args = parser.parse_args()
    loss_list = []
    acc_list = []

    # loading meta data
    meta_data_path = os.path.join(args.datadir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    videos = [x[0] for x in meta_data]

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(videos,
                                                  test_size=1-config.train_ratio)

    # opem lmdb
    db = lmdb.open(args.datadir+'.lmdb', readonly=True, map_size=int(50e9))

    # define transforms
    random_crop_size = config.instance_size - 2 * config.total_stride
    train_z_transforms = transforms.Compose([
        transforms.CenterCrop((config.exemplar_size, config.exemplar_size)),
        transforms.ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        # transforms.RandomCrop((random_crop_size, random_crop_size), config.max_translate),
        transforms.ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        transforms.CenterCrop((config.exemplar_size, config.exemplar_size)),
        transforms.ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # create dataset
    train_dataset = ImagnetVIDDataset(db, train_videos, args.datadir,
                                      train_z_transforms, train_x_transforms)
    valid_dataset = ImagnetVIDDataset(db, valid_videos, args.datadir,
                                      valid_z_transforms, valid_x_transforms, training=False)

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
    model = model.cuda()
    print("Using", torch.cuda.device_count(), "GPUs")
    print("Model running on GPU:", next(model.parameters()).is_cuda)
    model.init_weights()
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    for epoch in range(config.start_epoch, config.epoch):

        # model.train() tells your model that you are training the model.
        # So effectively layers like dropout, batchnorm etc. which behave
        # different on the train and test procedures
        model.train()

        running_loss = 0.0
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

            outputs = model((reference_var, search_var))
            loss = model.loss(outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


if __name__ == '__main__':
    main()
