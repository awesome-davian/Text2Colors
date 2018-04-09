import torch.utils.data as data
import torch
import pickle, os
import numpy as np
from skimage.color import rgb2lab
import warnings


class Dataset(data.Dataset):
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


def get_loader(batch_size, input_dict):
    train_src_path = os.path.join('./data/hexcolor_vf/train_names.pkl')
    train_trg_path = os.path.join('./data/hexcolor_vf/train_palettes_rgb.pkl')
    val_src_path = os.path.join('./data/hexcolor_vf/test_names.pkl')
    val_trg_path = os.path.join('./data/hexcolor_vf/test_palettes_rgb.pkl')

    train_dataset = Dataset(train_src_path, train_trg_path, input_dict)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               drop_last=True,
                                               shuffle=True)

    val_dataset = Dataset(val_src_path, val_trg_path, input_dict)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=2,
                                             drop_last=True,
                                             shuffle=False)

    return train_loader, val_loader
