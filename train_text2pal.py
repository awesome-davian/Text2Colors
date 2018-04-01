#! /usr/bin/env python
import os, argparse
import torch
from torch import cuda

from text2pal.model import *
from text2pal.data_loader import get_loader
from text2pal.train import *
from text2pal.embedding import *

parser = argparse.ArgumentParser(description='Interactive Colorization through Text')

parser.add_argument('--hidden_size', type=int, default=150)
parser.add_argument('--n_layers', type=int, default=4)

parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs for train')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--dropout_p', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--lambda_sL1', type=float, default=100.0, help='weight for L1 loss')
parser.add_argument('--lambda_KL', type=float, default=0.5, help='weight for KL loss')
parser.add_argument('--log_interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('--test_interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('--save_interval', type=int, default=100, help='how many steps to wait before saving [default:500]')
parser.add_argument('--save_dir', type=str, default='./text2pal_newCA2/models', help='where to save the trained models')

parser.add_argument('--loss_combination', type=str, default='att_test_')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, default='cnn_1_100.pkl')
args = parser.parse_args()


try:
    os.makedirs(os.path.join(args.save_dir, args.loss_combination,
                             'sL1'+str(args.lambda_sL1)+'_KL'+str(args.lambda_KL)))
except OSError:
    pass

cuda.set_device(args.gpu)
print("Running on GPU : ", args.gpu)

input_dict = prepare_data()
emb_file = os.path.join('./data', 'Color-Hex-vf.pth')

if os.path.isfile(emb_file):
    W_emb = torch.load(emb_file)
else:
    W_emb = load_pretrained_embedding(input_dict.word2index,
                                      '../data/glove.840B.300d.txt',
                                      300)
    W_emb = torch.from_numpy(W_emb)
    torch.save(W_emb, emb_file)

W_emb = W_emb.cuda()

encoder = EncoderRNN(input_dict.n_words, args.hidden_size,
                     args.n_layers, args.dropout_p, W_emb).cuda()
decoder = AttnDecoderRNN(args.hidden_size, input_dict,
                         args.n_layers, args.dropout_p).cuda()
discriminator = Discriminator(15, args.hidden_size).cuda()

train_loader, val_loader = get_loader(args.batch_size, input_dict)

print("Begin training...")
try:
    TrainGAN(train_loader, val_loader, encoder, decoder, discriminator, args).train()
except KeyboardInterrupt:
    print('-' * 80)
    print('Exiting from training early')