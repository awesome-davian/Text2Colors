import os
import numpy as np
from skimage.color import rgb2lab
from skimage import io

import torch
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from pal2color.global_hint import *

import re
import pickle

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class Dataset(data.Dataset):

    def __init__(self, root_dir, pal_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = os.listdir(root_dir)
        self.img_list.sort(key=natural_keys)
        self.palette_dir = pal_dir
        self.data = rgb2lab(np.load(self.palette_dir)
                            .reshape(-1,5,3)/255, 
                            illuminant='D50')
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_list[idx])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return (image, self.data[idx])


class LoadImagenet(data.Dataset):

    def __init__(self, image_dir, pal_dir):

        with open(image_dir,'rb') as f:
            self.image_data = np.asarray(pickle.load(f)) / 255

        with open(pal_dir,'rb') as f:
            self.pal_data = rgb2lab(np.asarray(pickle.load(f))
                                    .reshape(-1,5,3) / 256
                                    ,illuminant='D50')
                                      
        self.data_size = self.image_data.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.image_data[idx], self.pal_data[idx]


def Color_Dataloader(dataset, batch_size, idx=0):

    if dataset == 'imagenet':

        traindir = './data/imagenet/train_palette_set_origin/train_images_%d.txt' % (idx)
        pal_traindir = './data/imagenet/train_palette_set_origin/train_palette_%d.txt' % (idx)
        
        train_dataset = LoadImagenet(traindir, pal_traindir)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        imsize = 256

    elif dataset == 'bird256':

        traindir = './data/bird256/train_palette/train_images_origin.txt'
        pal_traindir = './data/bird256/train_palette/train_palette_origin.txt'
        
        train_dataset = LoadImagenet(traindir, pal_traindir)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        imsize = 256


    return (train_dataset, train_loader, imsize)


def process_palette_ab(pal_data, batch_size):

    img_a_scale = (pal_data[:, :, 1:2] + 88) / 185
    img_b_scale = (pal_data[:, :, 2:3] + 127) / 212
    img_ab_scale = np.concatenate((img_a_scale,img_b_scale),axis=2)
    ab_for_global = torch.from_numpy(img_ab_scale).float()
    ab_for_global = ab_for_global.view(batch_size, 10).unsqueeze(2).unsqueeze(2)

    return ab_for_global

def process_palette_lab(pal_data, batch_size):

    img_l = pal_data[:, :, 0:1] / 100
    img_a_scale = (pal_data[:, :, 1:2] + 88) / 185
    img_b_scale = (pal_data[:, :, 2:3] + 127) / 212
    img_lab_scale = np.concatenate((img_l, img_a_scale, img_b_scale),axis=2)
    lab_for_global = torch.from_numpy(img_lab_scale).float()
    lab_for_global = lab_for_global.view(batch_size, 15).unsqueeze(2).unsqueeze(2)

    return lab_for_global

def process_data(image_data, batch_size, imsize):
    input = torch.zeros(batch_size, 1, imsize, imsize)
    labels = torch.zeros(batch_size, 2, imsize, imsize)
    images_np = image_data.numpy().transpose((0, 2, 3, 1))

    for k in range(batch_size):

        img_lab = rgb2lab(images_np[k], illuminant='D50')
        img_l = img_lab[:, :,0] / 100
        input[k] = torch.from_numpy(np.expand_dims(img_l, 0))

        img_a_scale = (img_lab[:, :, 1:2] + 88) / 185
        img_b_scale = (img_lab[:, :, 2:3] + 127) / 212

        img_ab_scale = np.concatenate((img_a_scale,img_b_scale),axis=2)
        labels[k] = torch.from_numpy(img_ab_scale.transpose((2, 0, 1)))

    return input, labels