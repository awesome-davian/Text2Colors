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
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
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
                            illuminant='D50') # 9400 x 15 => 9400 x 5 x 3
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
                                    .reshape(-1,5,3) / 256 # 15 => 5 x 3,
                                    ,illuminant='D50')
                                      
        self.data_size = self.image_data.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.image_data[idx], self.pal_data[idx]


class LoadTemp(data.Dataset):

    def __init__(self, image_dir, pal_dir):

        with open(image_dir,'rb') as f:
            self.image_data = np.asarray(pickle.load(f)) / 255

        with open(pal_dir,'rb') as f:
            self.pal_data = rgb2lab(np.asarray(pickle.load(f))
                                    .reshape(-1,5,3) / 256 # 15 => 5 x 3,
                                    ,illuminant='D50')
                                      
        self.data_size = self.image_data.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.image_data[idx], self.pal_data[idx]


def Color_Dataloader(dataset, batch_size, idx=0):
    if dataset == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = dsets.CIFAR10(root='./data/',
                                      train=True,
                                      transform=transform,
                                      download=True)
        val_dataset = dsets.CIFAR10(root='./data/',
                                     train=False,
                                     transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        imsize = 32

    elif dataset == 'bird':

        traindir = './data/bird/train/'
        valdir = './data/bird/val/'
        testdir = './data/bird/test/'

        pal_traindir = './data/bird/rgb_train_palette/train_palette.npy'
        pal_valdir = './data/bird/rgb_val_palette/val_palette.npy'
        pal_testdir = './data/bird/rgb_test_palette/test_palette.npy'

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = Dataset(traindir, pal_traindir, transform=transform)
        val_dataset = Dataset(valdir, pal_valdir, transform=transform)
        test_dataset = Dataset(testdir, pal_testdir, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        imsize = 64

    elif dataset == 'flower':
        traindir = './data/flower/train/'
        valdir = './data/flower/val/'
        testdir = './data/flower/test/'

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = Dataset(traindir, transform=transform)
        val_dataset = Dataset(valdir, transform=transform)
        test_dataset = Dataset(testdir, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        imsize = 64

    elif dataset == 'imagenet':
        '''
        Downsampled dataset containing exactly the same 
        number of images as the original ImageNet, i.e., 
        1281167 training images from 1000 classes and 
        50000 validation images with 50 images per class.
        '''

        traindir = './data/imagenet/train_palette_set_origin/train_images_%d.txt' % (idx)
        pal_traindir = './data/imagenet/train_palette_set_origin/train_palette_%d.txt' % (idx)
        
        train_dataset = LoadImagenet(traindir, pal_traindir)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        imsize = 256

    elif dataset == 'bird256':
        '''
        Downsampled dataset containing exactly the same 
        number of images as the original ImageNet, i.e., 
        1281167 training images from 1000 classes and 
        50000 validation images with 50 images per class.
        '''

        traindir = './data/bird256/train_palette/train_images.txt'
        pal_traindir = './data/bird256/train_palette/train_palette.txt'
        
        train_dataset = LoadImagenet(traindir, pal_traindir)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        imsize = 256

    elif dataset == 'temp':
        '''
        Downsampled dataset containing exactly the same 
        number of images as the original ImageNet, i.e., 
        1281167 training images from 1000 classes and 
        50000 validation images with 50 images per class.
        '''

        traindir = './data/imagenet/train_palette/train_images_10000_j0.5.txt'
        pal_traindir = './data/imagenet/train_palette/train_palette_10000_j0.5.txt'
        
        train_dataset = LoadTemp(traindir, pal_traindir)
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

    # lab_for_global = torch.zeros(batch_size, 5, 3)
    # for k in range(batch_size):

    #     img_ab_unscale = image_data[k, :, 0:3]
    #     lab_for_global[k] = img_ab_unscale

    #     lab_for_global = lab_for_global.view(batch_size, 15).unsqueeze(2).unsqueeze(2)
    
    return lab_for_global

def process_data(image_data, batch_size, imsize):
    input = torch.zeros(batch_size, 1, imsize, imsize)
    labels = torch.zeros(batch_size, 2, imsize, imsize)
    images_np = image_data.numpy().transpose((0, 2, 3, 1))

    for k in range(batch_size):
        # images_np : 64 x 64 x 3

        img_lab = rgb2lab(images_np[k], illuminant='D50')
        # print(img_lab)
        img_l = img_lab[:, :,0] / 100
        input[k] = torch.from_numpy(np.expand_dims(img_l, 0))

        img_a_scale = (img_lab[:, :, 1:2] + 88) / 185
        img_b_scale = (img_lab[:, :, 2:3] + 127) / 212

        img_ab_scale = np.concatenate((img_a_scale,img_b_scale),axis=2)
        labels[k] = torch.from_numpy(img_ab_scale.transpose((2, 0, 1)))

    return input, labels