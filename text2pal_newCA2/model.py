import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from random import *

from text2pal_newCA2.utils import *


class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 150
        self.c_dim = 150
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))  # N x B x 600
        mu = x[:, :, :self.c_dim]                # N x B x 300
        logvar = x[:, :, self.c_dim:]           # N x B x 300
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0.0, 1)
        eps = Variable(eps)
        return eps * std + mu

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_p, W_emb=None):
        super(EncoderRNN, self).__init__()

        # Number of features in the hidden state h.
        self.hidden_size = hidden_size
        # Number of hidden layers (Number of recurrent layers).
        self.n_layers = n_layers

        self.embed = Embed(input_size, 300, W_emb, True)
        self.ca_net = CA_NET()
        self.gru = nn.GRU(300, hidden_size, n_layers, dropout=dropout_p)

    def forward(self, word_inputs, hidden):
        embedded = self.embed(word_inputs).transpose(0,1) # N x B x 300

        #c_code, mu, logvar = self.ca_net(embedded)  # N x B x 300
        output, hidden = self.gru(embedded, hidden)  # gru(N x B x 300, 2 x B x 150) -> N x B x 150, 2 x B x 150
        c_code, mu, logvar = self.ca_net(output)

        return c_code, hidden, mu, logvar

    def init_hidden(self,batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda() # n_layers x B x 150

        return hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, input_dict, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.input_dict = input_dict
        self.attn = Attn(hidden_size, input_dict.max_len)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.palette_dim = 3

        self.gru = nn.GRU(self.hidden_size + self.palette_dim, hidden_size, n_layers, dropout=dropout_p)

        self.out = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),
                        # nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.Linear(hidden_size,self.palette_dim)
                   )

    def forward(self, palette, last_context, last_hidden, encoder_outputs, each_input_size, signal):
        rnn_input = torch.cat((palette, last_context), 2) # palette: 1 x B x 3, last_context: 1 x B x 150
        
        rnn_output, hidden = self.gru(rnn_input, last_hidden) # rnn_output: 1 x B x 150, hidden: 4 x B x 150

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs, each_input_size) # rnn_output: 1 x B x 150 , encoder_outputs: N x B x 150
        
        # if signal:
        #     rand = randint(0,1)
        #     attn_weights = Variable(torch.zeros(attn_weights.size())).cuda()

        context = torch.bmm(attn_weights, encoder_outputs.transpose(0,1)) # (B x 1 x N) x (B x N x 150)

        rnn_output = rnn_output.squeeze(0) # 1 x B x 150 -> B x 150
        context = context.squeeze(1)       # B x 1 x 150 -> B x 150

        output = self.out(torch.cat((rnn_output, context), 1)) # B x 3
        # output = self.out(rnn_output) # B x 3
        # Return final output, hidden state, and attention weights (for visualization, B x 1 x N)
        return output, context.unsqueeze(0), hidden, attn_weights


class Attn(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)
        # self.attn_e = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_h = nn.Linear(self.hidden_size, self.hidden_size)
        # self.non_linear = nn.LeakyReLU()
        # self.non_linear = nn.Sigmoid()

    def forward(self, hidden, encoder_outputs, each_size): # hidden: B x 150, encoder_outputs: N x B x 150

        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len,batch_size,1)).cuda() # N x B x 1

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i]) # encoder output: N x B x H
        
        # -Inf
        for i in range(batch_size):
            
            try:
                attn_energies[each_size[i]:,i] = -float('Inf')
                # print(attn_energies[:each_size[i],i])
            except:
                pass # In case of the longest text

        attn_energies = self.softmax(attn_energies)

        # mask
        # attn_energies = mask(attn_energies, each_size, seq_len, batch_size)
        
        return attn_energies.permute(1,2,0) # Normalize energies to weights in range 0 to 1, resize to batch x 1 x N

    def score(self, hidden, encoder_output): # hidden : B x 150, encoder_output[i] : B x 150

        # encoder_ = self.attn_e(encoder_output)
        # A / ||A||
        encoder_ = encoder_ / (torch.sum((encoder_ ** 2), 1) ** 0.5).unsqueeze(1)
        encoder_ = encoder_output.unsqueeze(2)

        hidden_ = self.attn_h(hidden)
        # B / ||B||
        hidden_ = hidden_ / (torch.sum((hidden_ ** 2), 1) ** 0.5).unsqueeze(1)
        hidden_ = hidden_.unsqueeze(1)

        energy = torch.bmm(hidden_, encoder_)

        # energy = self.attn(torch.cat((hidden, encoder_output), 1)) # energy : B x 150
        # energy = torch.mm(energy,self.other) # (B x 150) X (150 x 1) = B x 1
        
        return energy.squeeze(2)

class Discriminator(nn.Module):
    def __init__(self, color_size=15, hidden_dim=150):
        super(Discriminator, self).__init__()

        curr_dim = color_size+hidden_dim

        layers = []
        layers.append(nn.Linear(curr_dim, int(curr_dim/2)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(int(curr_dim/2), int(curr_dim/4)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(int(curr_dim/4), int(curr_dim/8)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(int(curr_dim/8), 1)) # 9 -> 1
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, color, text):
        out = torch.cat([color, text], dim=1) # color: batch x 15, text: batch x 150
        out2 = self.main(out)
        return out2.squeeze(1)


def init_weights_normal(m):
    if type(m) == nn.Conv1d:
        m.weight.data.normal_(0.0, 0.05)
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.05)