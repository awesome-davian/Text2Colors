import torch
import numpy as np
from pal2color.model import *
from torch.autograd import Variable

def process_global_ab(input_ab, batch_size, always_give_global_hint):
    X_hist = input_ab

    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1 ,1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1 ,1))
        for l in range(batch_size):
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(10)

    
    global_input = torch.cat([X_hist, B_hist], 1)

    return global_input

def process_global_lab(input_lab, batch_size, always_give_global_hint):
    
    X_hist = input_lab
    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1 ,1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1 ,1))
        for l in range(batch_size):
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(15)

    global_input = torch.cat([X_hist, B_hist], 1)

    return global_input

def process_global_sampling_ab(palette, batch_size, imsize, hist_mean, hist_std):

    X_hist = palette
    B_hist = torch.ones(batch_size, 1, 1, 1)

    X_hist = Variable(X_hist).cuda()
    B_hist = Variable(B_hist).cuda()
    
    global_input = torch.cat([X_hist, B_hist], 1)

    return global_input

def process_global_sampling_lab(palette, batch_size, imsize, hist_mean, hist_std):

    X_hist = palette
    B_hist = torch.ones(batch_size, 1, 1, 1)

    X_hist = Variable(X_hist).cuda()
    B_hist = Variable(B_hist).cuda()
    
    global_input = torch.cat([X_hist, B_hist], 1)

    return global_input
