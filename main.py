#! /usr/bin/env python
from __future__ import division
import os
import argparse
from solver import Solver


def main(args):

    # Create directory if it doesn't exist.
    if not os.path.exists(args.text2pal_dir):
        os.makedirs(args.text2pal_dir)
    if not os.path.exists(args.pal2color_dir):
        os.makedirs(args.pal2color_dir)
    if not os.path.exists(args.train_sample_dir):
        os.makedirs(args.train_sample_dir)
    if not os.path.exists(os.path.join(args.test_sample_dir, args.mode)):
        os.makedirs(os.path.join(args.test_sample_dir, args.mode))

    # Solver for training and testing Text2Colors.
    solver = Solver(args)

    # Train or test.
    if args.mode == 'train_TPN':
        solver.train_TPN()

    elif args.mode == 'train_PCN':
        solver.train_PCN()

    elif args.mode == 'test_TPN':
        solver.test_TPN()

    elif args.mode == 'test_text2colors':
        solver.test_text2colors()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    # text2pal
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--n_layers', type=int, default=1)
    # pal2color
    parser.add_argument('--always_give_global_hint', type=int, default=1)
    parser.add_argument('--add_L', type=int, default=1)

    # Training and testing configuration.
    parser.add_argument('--mode', type=str, default='train_TPN',
                        choices=['train_TPN', 'train_PCN', 'test_TPN', 'test_text2colors'])
    parser.add_argument('--dataset', type=str, default='bird256', choices=['imagenet', 'bird256'])
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs for training')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--lambda_sL1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_KL', type=float, default=0.5, help='weight for KL loss')
    parser.add_argument('--lambda_GAN', type=float, default=0.1)

    # Directories.
    parser.add_argument('--text2pal_dir', type=str, default='./models/TPN')
    parser.add_argument('--pal2color_dir', type=str, default='./models/PCN')
    parser.add_argument('--train_sample_dir', type=str, default='./samples/train')
    parser.add_argument('--test_sample_dir', type=str, default='./samples/test')

    # Step size.
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many steps to wait before logging training status')
    parser.add_argument('--sample_interval', type=int, default=20,
                        help='how many steps to wait before saving the training output')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='how many steps to wait before saving the trained models')
    args = parser.parse_args()
    print(args)
    main(args)