class GanModel(nn.Module):

    def init(self, input_variable, target_variable, encoder, decoder, D, each_input_size, maxlen):

        self.fake_palette = None
        self.real_text = None
        self.true = None
        self.false = None
        self.attention = None # for visualization

        self.input_variable = input_variable
        self.real_palette = target_variable
        #Generator
        self.encoder = encoder
        self.decoder = decoder
        #Discriminator
        self.D = D
        self.each_input_size = each_input_size
        self.maxlen = maxlen

    def g_forward(self):

        # Get size of batch_size
        batch_size = self.input_variable.size(0)
        # Run words through encoder
        encoder_hidden = self.encoder.init_hidden()
        hidden_size = encoder_hidden.size(2)

        encoder_outputs, decoder_hidden = self.encoder(self.input_variable, encoder_hidden) # input_variable : B x N, encoder_hidden : 2 x B x 150
        #encoder_outpus: N x B x 150, decoder_hidden: 2 x B x 150

        # Prepare input and output variables
        decoder_context = Variable(torch.zeros(1,batch_size,self.decoder.hidden_size)).cuda() # 1 x B x 150

        # rand is uniform distribution [0,1)
        palette = Variable(torch.FloatTensor([[0]])).unsqueeze(2).expand(1,batch_size,3).cuda()
        palettes = Variable(torch.FloatTensor(batch_size,15).zero_()).cuda()
        decoder_attentions = Variable(torch.zeros(5, self.maxlen),requires_grad=False).cuda()

        # predict 5 color palette
        for i in range(5):
            palette, decoder_context ,decoder_hidden, decoder_attention = self.decoder(palette, decoder_context,
                                                                            decoder_hidden, encoder_outputs, self.each_input_size)
            palettes[:,3*i:3*(i+1)] = palette # B x 15[:,0~2] = B x 3
            palette = palette.unsqueeze(0)
            decoder_context = decoder_context.unsqueeze(0)
            # decoder_attention: B x 1 x N
            decoder_attentions[i] = decoder_attention[0].data

        each_input_size = Variable(torch.FloatTensor(self.each_input_size).unsqueeze(1)
                                    .expand(batch_size,hidden_size), # FC encoder = 300, RNN encoder = 150
                                    requires_grad=False).cuda() # B x 150

        # to average hidden states of each encoder module
        encoder_outputs = torch.sum(encoder_outputs,0)
        encoder_outputs = torch.div(encoder_outputs,each_input_size)

        # apply the hungarian algorithm to remove a sequential info
        palettes = hungarian(palettes, self.real_palette)

        self.fake_palette = palettes
        self.real_text = encoder_outputs
        self.attention = decoder_attentions.cpu().data

    def d_forward(self, isG):
        true = None
        if not isG:
            # real text real image
            true = self.D(self.real_palette, self.real_text) # target_variable : B x 15 # encoder_outputs : B x 150
        # real text fake image
        false = self.D(self.fake_palette, self.real_text) #

        self.true = true
        self.false = false

    #lc = gan loss coefficient
    def d_backward(self, D_optimizer, bce_criterion, isTrain, gan_loss):

        batch_size =  self.input_variable.size()[0]
        y_ones, y_zeros = (Variable(torch.ones(batch_size), requires_grad=False).cuda(),
                            Variable(torch.zeros(batch_size), requires_grad=False).cuda())

        # real text real image
        real_loss = bce_criterion(self.true, y_ones) # target_variable : B x 15 # encoder_outputs : B x 150
        # real text fake image
        fake_loss = bce_criterion(self.false, y_zeros) #
        D_loss = real_loss + fake_loss
        loss = gan_loss * D_loss

        if isTrain:
            D_optimizer.zero_grad()
            loss.backward()
            D_optimizer.step()

        return loss.data[0]

    def g_backward(self, encoder_optimizer, decoder_optimizer, D_optimizer,
                    bce_criterion, criterion, isTrain, gan_loss):

        batch_size = self.input_variable.size()[0]
        G_loss = Variable(torch.zeros(1)).cuda()
        if gan_loss > 0:
            y_ones = Variable(torch.ones(batch_size), requires_grad=False).cuda()
            G_loss = gan_loss * bce_criterion(self.false, y_ones)
        MSEloss = criterion(self.fake_palette, self.real_palette)
        loss = MSEloss + G_loss
        # loss = MSEloss
        if isTrain:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            D_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            D_optimizer.step()

        return MSEloss.data[0], G_loss.data[0]
        # return MSEloss.data[0]

    def getPalette(self):
        if self.real_palette is not None:
            return self.fake_palette.clone(), self.real_palette.clone()
        return self.fake_palette.clone()

    def getAttention(self):
        return self.attention