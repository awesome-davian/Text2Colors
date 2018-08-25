import torch
import torch.utils.data as data
import pickle
import os
import numpy as np
from skimage.color import rgb2lab
import warnings

class PAT_Dataset(data.Dataset):
    def __init__(self, src_path, trg_path, input_dict):
        with open(src_path, 'rb') as fin:
            self.src_seqs = pickle.load(fin)
        with open(trg_path, 'rb') as fin:
            self.trg_seqs = pickle.load(fin)

        words_index = []
        for index, palette_name in enumerate(self.src_seqs):
            temp = [0] * input_dict.max_len

            for i, word in enumerate(palette_name):
                temp[i] = input_dict.word2index[word]
            words_index.append(temp)
        self.src_seqs = torch.LongTensor(words_index)

        palette_list = []
        for index, palettes in enumerate(self.trg_seqs):
            temp = []
            for palette in palettes:
                rgb = np.array([palette[0], palette[1], palette[2]]) / 255.0
                warnings.filterwarnings("ignore")
                lab = rgb2lab(rgb[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
                temp.append(lab[0])
                temp.append(lab[1])
                temp.append(lab[2])
            palette_list.append(temp)

        self.trg_seqs = torch.FloatTensor(palette_list)
        self.num_total_seqs = len(self.src_seqs)

    def __getitem__(self, index):
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_seqs


def t2p_loader(batch_size, input_dict):
    train_src_path = os.path.join('./data/hexcolor_vf/train_names.pkl')
    train_trg_path = os.path.join('./data/hexcolor_vf/train_palettes_rgb.pkl')
    val_src_path = os.path.join('./data/hexcolor_vf/test_names.pkl')
    val_trg_path = os.path.join('./data/hexcolor_vf/test_palettes_rgb.pkl')

    train_dataset = PAT_Dataset(train_src_path, train_trg_path, input_dict)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               drop_last=True,
                                               shuffle=True)

    test_dataset = PAT_Dataset(val_src_path, val_trg_path, input_dict)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=batch_size,
                                             num_workers=2,
                                             drop_last=True,
                                             shuffle=False)

    return train_loader, test_loader


class Image_Dataset(data.Dataset):
    def __init__(self, image_dir, pal_dir):
        with open(image_dir, 'rb') as f:
            self.image_data = np.asarray(pickle.load(f)) / 255

        with open(pal_dir, 'rb') as f:
            self.pal_data = rgb2lab(np.asarray(pickle.load(f))
                                    .reshape(-1, 5, 3) / 256
                                    , illuminant='D50')

        self.data_size = self.image_data.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.image_data[idx], self.pal_data[idx]


def p2c_loader(dataset, batch_size, idx=0):
    if dataset == 'imagenet':

        train_img_path = './data/imagenet/train_palette_set_origin/train_images_%d.txt' % (idx)
        train_pal_path = './data/imagenet/train_palette_set_origin/train_palette_%d.txt' % (idx)

        train_dataset = Image_Dataset(train_img_path, train_pal_path)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        imsize = 256

    elif dataset == 'bird256':

        train_img_path = './data/bird256/train_palette/train_images_origin.txt'
        train_pal_path = './data/bird256/train_palette/train_palette_origin.txt'

        train_dataset = Image_Dataset(train_img_path, train_pal_path)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        imsize = 256

    return train_loader, imsize


class Test_Dataset(data.Dataset):
    def __init__(self, input_dict, txt_path, pal_path, img_path, transform=None):
        self.transform = transform
        with open(img_path, 'rb') as f:
            self.images = np.asarray(pickle.load(f)) / 255
        with open(txt_path, 'rb') as fin:
            self.src_seqs = pickle.load(fin)
        with open(pal_path, 'rb') as fin:
            self.trg_seqs = pickle.load(fin)

        # ==================== Preprocessing src_seqs ====================#
        # Return a list of indexes, one for each word in the sentence.
        words_index = []
        for index, palette_name in enumerate(self.src_seqs):
            # Set list size to the longest palette name.
            temp = [0] * input_dict.max_len
            for i, word in enumerate(palette_name):
                temp[i] = input_dict.word2index[word]
            words_index.append(temp)

        self.src_seqs = torch.LongTensor(words_index)

        # ==================== Preprocessing trg_seqs ====================#
        palette_list = []
        for palettes in self.trg_seqs:
            temp = []
            for palette in palettes:
                rgb = np.array([palette[0], palette[1], palette[2]]) / 255.0
                warnings.filterwarnings("ignore")
                lab = rgb2lab(rgb[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
                temp.append(lab[0])
                temp.append(lab[1])
                temp.append(lab[2])
            palette_list.append(temp)

        self.trg_seqs = torch.FloatTensor(palette_list)

        self.num_total_data = len(self.src_seqs)

    def __len__(self):
        return self.num_total_data

    def __getitem__(self, idx):
        """Returns one data pair."""
        text = self.src_seqs[idx]
        palette = self.trg_seqs[idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        return text, palette, image


def test_loader(dataset, batch_size, input_dict):

    if dataset == 'bird256':

        txt_path = './data/hexcolor_vf/test_names.pkl'
        pal_path = './data/hexcolor_vf/test_palettes_rgb.pkl'
        img_path = './data/bird256/test_palette/test_images_origin.txt'

        test_dataset = Test_Dataset(input_dict, txt_path, pal_path, img_path)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)
        imsize = 256

    return test_loader, imsize