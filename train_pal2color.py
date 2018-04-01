from __future__ import division
import os
import torch
import argparse
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable

from pal2color.model import *
from pal2color.util import *
from pal2color.global_hint import *
from pal2color.data_loader import *
from pal2color.gan import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bird256', choices=['imagenet','bird256'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='./pal2color/models/')
    parser.add_argument('--log_path', type=str, default='./pal2color/logs')
    parser.add_argument('--image_save', type=str, default='./pal2color/images')
    parser.add_argument('--learning_rate', type=int, default=0.0002)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--dropout_p', type=int, default=0.2)
    parser.add_argument('--resume', type=bool, default=False,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--gan_loss', type=float, default=0.1)

    parser.add_argument('--always_give_global_hint', type=int, default=1)
    parser.add_argument('--multi_injection', type=int, default=1)
    parser.add_argument('--add_L', type=int, default=1)
    return parser.parse_args()


def main(args):
    dataset = args.data
    gpu = args.gpu
    batch_size = args.batch_size
    dropout_p = args.dropout_p
    model_path = args.model_path
    log_path = args.log_path
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    start_epoch = args.start_epoch
    gan_loss = args.gan_loss
    always_give_global_hint = args.always_give_global_hint
    multi_injection = args.multi_injection
    add_L = args.add_L

    print("Running on gpu : ", gpu)
    cuda.set_device(gpu)

    make_folder(model_path, dataset)
    make_folder(log_path, dataset +'/ckpt')

    (train_dataset, train_loader, imsize) = Color_Dataloader(dataset, batch_size, 0)
    (G, D, G_optimizer, D_optimizer, G_scheduler, D_scheduler) = init_models(batch_size, imsize, dropout_p, learning_rate, multi_injection, add_L)
        
    criterion_sL1 = nn.SmoothL1Loss().cuda()
    criterion_bce = nn.BCELoss().cuda()

    (G, G_optimizer, D, D_optimizer, _, start_epoch) = resume(args.resume, log_path, dataset, G, G_optimizer, D, D_optimizer)

    tell_time = Timer()
    iter = 0
    gm = GanModel()

    for epoch in range(start_epoch, num_epochs):

        G.train()
        for i, (images, pals) in enumerate(train_loader):

            (_, _, loss, sL1_loss) = train(gm, images, pals, G, D, G_optimizer, D_optimizer,
                                            criterion_bce, criterion_sL1, always_give_global_hint, 
                                            add_L, gan_loss, True)

            num_batches = (len(train_dataset) // batch_size)
            print_log(0, 0, epoch, i, num_epochs, num_batches, sL1_loss, tell_time, iter)

        checkpoint = {
            'epoch': epoch + 1,
            'args': args,
            'G_state_dict': G.state_dict(),
            'G_optimizer': G_optimizer.state_dict(),
            'D_state_dict': D.state_dict(),
            'D_optimizer': D_optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(log_path, dataset, 'ckpt/model.ckpt'))
        msg = "epoch: %d" % (epoch)
        if (epoch + 1) % 10 == 0:
            print ('Saved model')
            torch.save(G.state_dict(), os.path.join(
                model_path, dataset, '%s_cGAN-unet_bird256.pkl' % (msg)))


if __name__ == '__main__':
    args = parse_args()
    main(args)