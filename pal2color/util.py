import os
import numpy as np
import time
import datetime
import torch
import warnings
from skimage.color import rgb2lab, lab2rgb, rgb2gray

def check_value(inds, val):
    if (np.array(inds).size == 1):
        if (inds == val):
            return True
    return False

def flatten_nd_array(pts_nd, axis=1):

    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS, SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):

    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))
    NPTS = np.prod(SHP[nax])

    if (squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

def na():
    return np.newaxis


class Timer():
    def __init__(self):
        self.cur_t = time.time()

    def tic(self):
        self.cur_t = time.time()

    def toc(self):
        return time.time() - self.cur_t

    def tocStr(self, t=-1):
        if (t == -1):
            return str(datetime.timedelta(seconds=np.round(time.time() - self.cur_t, 3)))[:-4]
        else:
            return str(datetime.timedelta(seconds=np.round(t, 3)))[:-4]

def distribution(tensor):

	tensor = torch.div(tensor, expand(tensor.sum(dim=1).unsqueeze(-1), tensor))
	if (tensor.sum(dim=1).data.cpu().numpy()==0).any():
		print ("")
		print ("")
		print ("division by zero")
		print ("")
		print ("")
	return tensor.unsqueeze(-1)


def make_folder(path, dataset):
    try:
        os.makedirs(os.path.join(path, dataset))
    except OSError:
        pass

def num_param(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def print_log(idx, num_idx, epoch, mini_batch, num_epochs, num_batches, sL1_loss, tell_time, iter):
    if (mini_batch + 1) % 10 == 0:
        print('Epoch [%d/%d], IDX [%d/%d], Iter [%d/%d], sL1_loss: %.10f, iter_time: %2.2f, aggregate_time: %6.2f'
              % (epoch + 1, num_epochs, idx, num_idx, mini_batch + 1, num_batches, sL1_loss,
                 (tell_time.toc() - iter), tell_time.toc()))
        iter = tell_time.toc()

def resume(resume_, log_path, dataset, G, G_optimizer, D, D_optimizer):
    start_idx=1
    start_epoch=0
    if resume_:
        ckpt_path = os.path.join(log_path, dataset, 'ckpt/model_origin.ckpt')
        if os.path.isfile(ckpt_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(ckpt_path)
            try:
                start_idx = checkpoint['idx']
            except:
                start_idx = 0
            start_epoch = checkpoint['epoch']
            G.load_state_dict(checkpoint['G_state_dict'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            D.load_state_dict(checkpoint['D_state_dict'])
            D_optimizer.load_state_dict(checkpoint['D_optimizer'])
            print("Start training from epoch {}.".format(checkpoint['epoch']+1))
        else:
            print("Sorry, no checkpoint found.")

    return G, G_optimizer, D, D_optimizer, start_idx, start_epoch

def lab2rgb_1d(in_lab, clip=True):
    warnings.filterwarnings("ignore")
    tmp_rgb = lab2rgb(in_lab[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    return tmp_rgb