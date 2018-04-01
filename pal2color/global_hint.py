import torch
import numpy as np
from skimage.color import rgb2hsv
from pal2color.model import *
from torch.autograd import Variable
class Quantization():
    # Encode points as a linear combination of unordered points
	# using NN search and RBF kernel
    def __init__(self,batch, imsize, km_filepath='./data/pts_in_hull.npy' ):
        self.cc = torch.from_numpy(np.load(km_filepath)).type(torch.FloatTensor) # 313 x 2
        self.K = self.cc.shape[0]
        self.batch = batch
        self.imsize = imsize

    def encode_nn(self,images): # batch x imsize x imsize x 2
        images = images.permute(0,2,3,1) # batch x 2 x imsize x imsize -> batch x imsize x imsize x 2
        images_flt = images.contiguous().view(-1, 2)
        P = images_flt.shape[0]
        inds = self.nearest_inds(images_flt, self.cc).unsqueeze(1) # P x 1
        images_encoded = torch.zeros(P, self.K)
        images_encoded.scatter_(1, inds, 1)
        images_encoded = images_encoded.view(self.batch, self.imsize, self.imsize, 313)
        images_encoded = images_encoded.permute(0,3,1,2)
        return images_encoded

    def encode_nn_pal(self,palettes): # batch x 2 x 5
        palettes = palettes.permute(0,2,3,1) # batch x 2 x 5 x 5 -> batch x 5 x 5 x 2
        palettes_flt = palettes.contiguous().view(-1, 2)
        P = palettes_flt.shape[0]
        inds = self.nearest_inds(palettes_flt, self.cc).unsqueeze(1) # P x 1
        inds[0][0] = 310
        images_encoded = torch.zeros(P, self.K)                      # P x 313
        images_encoded.scatter_(1, inds, 1)
        images_encoded = images_encoded.view(self.batch, 5, 5, 313)
        images_encoded = images_encoded.permute(0,3,1,2)               # B x 313 x 5 x 5
        return images_encoded

    def nearest_inds(self, x, y):
        inner = torch.matmul(x, y.t())  # x = n x 2, y = 313 x 2, inner = n x 313
        normX = torch.sum(torch.mul(x, x), 1).unsqueeze(1).expand_as(inner)
        normY = torch.sum(torch.mul(y, y), 1).unsqueeze(1).t().expand_as(inner)  # n x 313
        P = normX - 2 * inner + normY
        nearest_idx = torch.min(P, dim=1)[1]
        return nearest_idx


class Global_Quant():
    ''' Layer which encodes ab map into Q colors'''
    def __init__(self, batch, imsize):
        self.quantization = Quantization(batch, imsize, km_filepath='./data/pts_in_hull.npy')

    def global_histogram(self, input):
        out = self.quantization.encode_nn(input) # batch x 313 x imsize x imsize
        out = out.type(torch.FloatTensor)        # change it to tensor
        X_onehotsum = torch.sum(torch.sum(out, dim=3), dim=2) # sum it up to batch x 313
        X_sum = torch.sum(X_onehotsum, dim=1).unsqueeze(1).expand_as(X_onehotsum)
        X_hist = torch.div(X_onehotsum, X_sum)   # make 313 probability
        return X_hist

    def global_histogram_pal(self, palette):           # palette: batch x 2 x 5 x 5
        out = self.quantization.encode_nn_pal(palette) # out: batch x 313 x 5 x 5
        out = out.type(torch.FloatTensor)                     # change it to tensor
        X_onehotsum = torch.sum(torch.sum(out, dim=3), dim=2) # batch x 313
        X_sum = torch.sum(X_onehotsum, dim=1).unsqueeze(1).expand_as(X_onehotsum)
        X_hist = torch.div(X_onehotsum, X_sum) # make 313 probability

        return X_hist

    def global_saturation(self, images): # input: tensor images batch x 3 x imsize x imsize (rgb)
        images_np = images.numpy().transpose((0, 2, 3, 1)) # numpy: batch x imsize x imsize x 3
        images_h = torch.zeros(images.size(0), 1, images.size(2),images.size(2))
        for k in range(images.size(0)):
            img_hsv = rgb2hsv(images_np[k])
            img_h = img_hsv[:, :, 1]
            images_h[k] = torch.from_numpy(img_h).unsqueeze(0) #  batch x 1 x imsize x imsize
        avgs = torch.mean(images_h.view(images.size(0), -1),dim=1) # batch
        return avgs

    def global_saturation_pal(self, palettes):  # palettes: batch x 3 x 5 x 5
        palettes_np = palettes.data.cpu().numpy().transpose((0, 2, 3, 1))  # batch x 5 x 5 x 3
        palettes_h = torch.zeros(palettes.size(0), 5, 5, 1)
        for k in range(palettes.size(0)):
            warnings.filterwarnings("ignore")
            img_rgb = lab2rgb(palettes_np[k], illuminant='D50')
            img_rgb = np.clip(img_rgb, 0, 1) * 255
            img_hsv = rgb2hsv(img_rgb)
            img_h = img_hsv[:, :, 1]
            palettes_h[k] = torch.from_numpy(img_h).unsqueeze(0) # 1 x 5 x 5 x 1
        avgs = torch.mean(palettes_h.view(palettes.size(0), -1),dim=1) # batch
        return avgs

    def global_masks(self, batch_size):  # both for histogram and saturation
        B_hist = torch.round(torch.rand(batch_size, 1))
        B_sat = torch.round(torch.rand(batch_size, 1))
        return B_hist, B_sat


def process_global_ab(input_ab, batch_size, always_give_global_hint):
    # glob_quant = Global_Quant(batch_size, imsize)
    X_hist = input_ab  # B x 10  batch x 313 x imsize x imsize

    # X_sat = glob_quant.global_saturation(images).unsqueeze(1)  # batch x 1
    # B_hist, B_sat = glob_quant.global_masks(batch_size)  # if masks are 0, put uniform random(0~1) value in it
    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1 ,1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1 ,1)) # B x 1 x 1 x 1
        for l in range(batch_size):
            # if B_sat[l].numpy() == 0:
                # X_sat[l] = torch.normal(torch.FloatTensor([hist_mean]), std=torch.FloatTensor([hist_std]))
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(10)

    # global_input = torch.cat([X_hist, B_hist, X_sat, B_sat], 1).unsqueeze(2).unsqueeze(2)
    global_input = torch.cat([X_hist, B_hist], 1)
    # batch x (q+1) = batch x 316 x 1 x 1

    return global_input # B x 11 x 1 x 1

def process_global_lab(input_lab, batch_size, always_give_global_hint):
    # glob_quant = Global_Quant(batch_size, imsize)
    X_hist = input_lab  # B x 10  batch x 313 x imsize x imsize
    
    # X_sat = glob_quant.global_saturation(images).unsqueeze(1)  # batch x 1
    # B_hist, B_sat = glob_quant.global_masks(batch_size)  # if masks are 0, put uniform random(0~1) value in it
    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1 ,1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1 ,1)) # B x 1 x 1 x 1
        for l in range(batch_size):
            # if B_sat[l].numpy() == 0:
                # X_sat[l] = torch.normal(torch.FloatTensor([hist_mean]), std=torch.FloatTensor([hist_std]))
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(15)

    # global_input = torch.cat([X_hist, B_hist, X_sat, B_sat], 1).unsqueeze(2).unsqueeze(2)
    global_input = torch.cat([X_hist, B_hist], 1)
    # batch x (q+1) = batch x 316 x 1 x 1

    return global_input # B x 16 x 1 x 1

# b x 10 x 1 x 1
def process_global_sampling_ab(palette, batch_size, imsize, hist_mean, hist_std):
    # glob_quant = Global_Quant(batch_size, imsize)
    # if HIST==True:
    X_hist = palette
    B_hist = torch.ones(batch_size, 1, 1, 1)

    X_hist = Variable(X_hist).cuda()
    B_hist = Variable(B_hist).cuda()
    # else:
    #     X_hist = torch.rand(batch_size, 10, 1, 1)
    #     B_hist = Variable(torch.zeros(batch_size, 1, 1, 1)).cuda()

    # if SAT==True:
    #     palette_sat = palette.view(-1, 3, 5).unsqueeze(3).repeat(1, 1, 1, 5)
    #     X_sat = glob_quant.global_saturation_pal(palette_sat).unsqueeze(1)  # batch x 1
    #     B_sat = torch.ones(batch_size, 1)

    # else: # if masks are 0, put uniform random(0~1) value in it
    #     X_sat = torch.randn(batch_size, 1)
    #     for l in range(batch_size):
    #         X_sat[l] = torch.normal(torch.FloatTensor([hist_mean]), std=torch.FloatTensor([hist_std]))
    #     B_sat = torch.zeros(batch_size, 1)

    # global_input = torch.cat([X_hist, B_hist, X_sat, B_sat], 1).unsqueeze(2).unsqueeze(2)
    global_input = torch.cat([X_hist, B_hist], 1)
    # batch x (q+1) = batch x 316 x 1 x 1

    return global_input

# b x 15 x 1 x 1
def process_global_sampling_lab(palette, batch_size, imsize, hist_mean, hist_std):
    # glob_quant = Global_Quant(batch_size, imsize)
    # if HIST==True:
    X_hist = palette # B x 15 x 1 x 1
    B_hist = torch.ones(batch_size, 1, 1, 1)

    X_hist = Variable(X_hist).cuda()
    B_hist = Variable(B_hist).cuda()
    # else:
        # X_hist = torch.rand(batch_size, 10, 1, 1)
        # B_hist = Variable(torch.zeros(batch_size, 1, 1, 1)).cuda()

    # if SAT==True:
    #     palette_sat = palette.view(-1, 3, 5).unsqueeze(3).repeat(1, 1, 1, 5)
    #     X_sat = glob_quant.global_saturation_pal(palette_sat).unsqueeze(1)  # batch x 1
    #     B_sat = torch.ones(batch_size, 1)

    # else: # if masks are 0, put uniform random(0~1) value in it
    #     X_sat = torch.randn(batch_size, 1)
    #     for l in range(batch_size):
    #         X_sat[l] = torch.normal(torch.FloatTensor([hist_mean]), std=torch.FloatTensor([hist_std]))
    #     B_sat = torch.zeros(batch_size, 1)

    # global_input = torch.cat([X_hist, B_hist, X_sat, B_sat], 1).unsqueeze(2).unsqueeze(2)
    global_input = torch.cat([X_hist, B_hist], 1)
    # batch x (q+1) = batch x 316 x 1 x 1

    return global_input


def kMeans(X, K, maxIters = 10, plot_progress = None):
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        if plot_progress != None: plot_progress(X, C, np.array(centroids))
    return np.array(centroids) , C