import torch
from torch.autograd import Variable
from .model import *
from .global_hint import *
from .data_loader import *

def train(gm, images, pals, G, D, G_optimizer, D_optimizer, criterion_bce, 
        criterion_sL1, always_give_global_hint, add_L, gan_loss=0.1, isTrain=True):

    D_loss = Variable(torch.zeros(1)).data
    gm.image_process(images, pals, add_L, always_give_global_hint)

    gm.init(G, D)
    gm.g_forward() # Run the train function; output => fake_palette, ground_truth => real palette
    gm.d_forward(True) # (real image, real text) => true, (fake image, real text) => false
    D_loss = gm.d_backward(D_optimizer, criterion_bce, isTrain, gan_loss)

    gm.init(G, D)
    gm.g_forward()
    gm.d_forward(False)
    sL1_loss, G_loss = gm.g_backward(G_optimizer, D_optimizer,
                    criterion_bce, criterion_sL1, isTrain, gan_loss)

    output, ground_truth = gm.getImage()

    loss = sL1_loss + G_loss + D_loss

    return output, ground_truth, loss, sL1_loss


class GanModel(nn.Module):

    def init(self, unet, discriminator):

        self.fake_image = None
        self.true = None
        self.false = None

        #Generator
        self.G = unet
        #Discriminator
        self.D = discriminator

    def image_process(self, images, pals, always_give_global_hint, add_L):
        
        batch = images.size(0)
        imsize = images.size(3)

        # (L)input: B x 1 x 64 x 64(normalized ab) labels: B x 2 x 64 x 64(unnorm ab)ab_for_global: B x 5 x 2
        inputs, labels = process_data(images, batch, imsize)
        if add_L:
            for_global = process_palette_lab(pals, batch)
            global_hint = process_global_lab(for_global, batch, always_give_global_hint)
        else:
            for_global = process_palette_ab(pals, batch)
            global_hint = process_global_ab(for_global, batch, always_give_global_hint)

        self.L_image = Variable(inputs).cuda() # input, L, (B x 1 x 64 x 64)
        self.real_image = Variable(labels).cuda() # ground_truth, ab, (B x 2 x 64 x 64)
        self.global_hint = Variable(global_hint).cuda() # (B x 15 or 10 x 1 x 1)

    def g_forward(self):

        batch = self.global_hint.size(0)
        self.fake_image = self.G(self.L_image, self.global_hint) # condition = global_hint

    def d_forward(self, isD):
        true = None
        imsize = self.real_image.size(3)
        global_hint = (self.global_hint).expand(-1,-1,imsize,imsize) # B x 10 or 15 x 1 x 1 => B x 10 or 15 x 64 x 64
        if isD:
            # real image
            true = self.D(torch.cat((self.real_image, global_hint), dim=1)) # target_variable : B x 15 # encoder_outputs : B x 150
        # fake image
        false = self.D(torch.cat((self.fake_image, global_hint), dim=1)) #

        self.true = true
        self.false = false

    #lc = gan loss coefficient
    def d_backward(self, D_optimizer, criterion_bce, isTrain, gan_loss):

        batch_size =  self.global_hint.size(0)
        y_ones, y_zeros = (Variable(torch.ones(batch_size, 1), requires_grad=False).cuda(),
                            Variable(torch.zeros(batch_size, 1), requires_grad=False).cuda())

        # real image
        real_loss = criterion_bce(self.true, y_ones) # target_variable : B x 15 # encoder_outputs : B x 150
        # fake image
        fake_loss = criterion_bce(self.false, y_zeros) #
        D_loss = real_loss + fake_loss
        loss = gan_loss * D_loss

        if isTrain:
            D_optimizer.zero_grad()
            loss.backward()
            D_optimizer.step()

        return loss.data[0]

    def g_backward(self, G_optimizer, D_optimizer,
                    criterion_bce, criterion_sL1, isTrain, gan_loss):

        batch_size = self.L_image.size(0)
        G_loss = Variable(torch.zeros(1)).cuda()
        y_ones = Variable(torch.ones(batch_size, 1), requires_grad=False).cuda()
        G_loss = gan_loss * criterion_bce(self.false, y_ones)

        # make outputs and labels as a matrix for loss calculation
        outputs = self.fake_image.view(batch_size, -1)             # 100 x 32*32*3(2048)
        labels = self.real_image.contiguous().view(batch_size, -1)  # 100 x 32*32*3

        sL1_loss = criterion_sL1(outputs, labels)
        loss = sL1_loss + G_loss

        if isTrain:
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()

        return sL1_loss.data[0], G_loss.data[0]

    def getImage(self):
        if self.real_image is not None:
            return self.fake_image.clone(), self.real_image.clone()
        return self.fake_image.clone()
