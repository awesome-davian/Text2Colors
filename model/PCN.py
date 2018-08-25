import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNetConvBlock1_1(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super(UNetConvBlock1_1, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class UNetConvBlock1_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock1_2, self).__init__()
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, 1, stride=2, groups=out_size, bias=False)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        out = self.conv3(out)
        return out


class UNetConvBlock1_2_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock1_2_2, self).__init__()
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(x)
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, 1, stride=2, groups=out_size, bias=False)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        out = self.conv3(out)
        return out


class UNetConvBlock2_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock2_2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock3, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.conv4 = nn.Conv2d(out_size, out_size, 1, stride=2, groups=out_size, bias=False)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        out = self.conv4(out)
        return out


class UNetConvBlock3_2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock3_2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock4(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock4, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock5(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock5, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock6(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock6, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=2, dilation=2)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock7(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock7, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock8(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetConvBlock8, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, dilation=1)
        self.bridge = nn.Conv2d(256, 256, kernel_size, padding=1)

        self.conv = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.activation(self.bridge(bridge) + up)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        out = self.batchnorm(out)
        return out


class UNetConvBlock9(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetConvBlock9, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, dilation=1)
        self.bridge = nn.Conv2d(128, 128, kernel_size, padding=1)
        self.conv = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.batchnorm = nn.BatchNorm2d(out_size)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.activation(self.bridge(bridge) + up)
        out = self.activation(self.conv(out))
        out = self.batchnorm(out)

        return out


class UNetConvBlock10(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetConvBlock10, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, dilation=1)
        self.bridge = nn.Conv2d(64, 128, kernel_size, padding=1)
        self.conv = nn.Conv2d(out_size, out_size, kernel_size, padding=1, dilation=1)
        self.activation = activation
        self.activation2 = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.activation(self.bridge(bridge) + up)
        out = self.activation2(self.conv(out))
        return out


class prediction(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, activation=F.sigmoid, space_dropout=False):
        super(prediction, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, dilation=1)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        return out


class convrelu(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, activation=F.relu, space_dropout=False):
        super(convrelu, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=0)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        return out


class global_network(nn.Module):
    def __init__(self, image_size, add_L):
        super(global_network, self).__init__()
        if add_L:
            self.oneD = convrelu(16, 128)
        else:
            self.oneD = convrelu(11, 128)
        self.twoD = convrelu(128, 256)
        self.threeD = convrelu(256, 512)
        self.fourD = convrelu(512, 512)
        self.image_size = image_size

    def forward(self, x, dim):
        n = 2
        out = self.oneD(x)

        if dim >= 256:
            n = 4
            out = self.twoD(out)
        if dim == 512:
            n = 8
            out = self.threeD(out)
            out = self.fourD(out)

        out = out.repeat(1, 1, int(self.image_size / n), int(self.image_size / n))
        return out


class UNet(nn.Module):
    def __init__(self, imsize, add_L):
        super(UNet, self).__init__()
        self.imsize = imsize
        self.globalnet512 = global_network(self.imsize, add_L)
        self.globalnet256 = global_network(self.imsize, add_L)
        self.globalnet128 = global_network(self.imsize, add_L)

        self.convlayer1_1 = UNetConvBlock1_1(1, 64)
        self.convlayer1_2 = UNetConvBlock1_2(64, 64)
        self.convlayer1_2_2 = UNetConvBlock1_2_2(64, 64)
        self.convlayer2 = UNetConvBlock2(64, 128)
        self.convlayer2_2 = UNetConvBlock2_2(64, 128)
        self.convlayer3 = UNetConvBlock3(128, 256)
        self.convlayer3_2 = UNetConvBlock3_2(128, 256)
        self.convlayer4 = UNetConvBlock4(256, 512)
        self.convlayer5 = UNetConvBlock5(512, 512)
        self.convlayer6 = UNetConvBlock6(512, 512)
        self.convlayer7 = UNetConvBlock7(512, 512)
        self.convlayer8 = UNetConvBlock8(512, 256)
        self.convlayer9 = UNetConvBlock9(256, 128)
        self.convlayer10 = UNetConvBlock10(128, 128)

        self.prediction = prediction(128, 2)

    def forward(self, x, side_input):
        layer1_1 = self.convlayer1_1(x)
        layer1_2 = self.convlayer1_2(layer1_1)
        layer1_2_2 = self.convlayer1_2_2(layer1_1)
        layer2 = self.convlayer2(layer1_2)
        layer2_2 = self.convlayer2_2(layer1_2)
        layer3 = self.convlayer3(layer2)
        layer3_2 = self.convlayer3_2(layer2)
        layer4 = self.convlayer4(layer3)

        global_net512 = self.globalnet512(side_input, 512)
        layer4 = layer4 + global_net512
        layer5 = self.convlayer5(layer4)
        layer6 = self.convlayer6(layer5)
        layer7 = self.convlayer7(layer6)

        layer8 = self.convlayer8(layer7, layer3_2)
        global_net256 = self.globalnet256(side_input, 256)
        layer8 = layer8 + global_net256

        layer9 = self.convlayer9(layer8, layer2_2)
        global_net128 = self.globalnet128(side_input, 128)
        layer9 = layer9 + global_net128

        layer10 = self.convlayer10(layer9, layer1_2_2)

        prediction = self.prediction(layer10)

        return prediction


class Discriminator(nn.Module):
    def __init__(self, add_L, imsize, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()

        input_dim = 2 + 10
        if add_L:
            input_dim = 3 + 15

        layers = []
        layers.append(nn.Conv2d(input_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(imsize / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(k_size * k_size * curr_dim),
            nn.Linear(k_size * k_size * curr_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        h = self.main(x)
        out = self.conv1(h)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out
