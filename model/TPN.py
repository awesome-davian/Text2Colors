import torch
import torch.nn as nn
from random import *
from util import *

class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 150
        self.c_dim = 150
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :, :self.c_dim]
        logvar = x[:, :, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0.0, 1)
        return eps * std + mu

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_p, W_emb=None):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embed = Embed(input_size, 300, W_emb, True)
        self.gru = nn.GRU(300, hidden_size, n_layers, dropout=dropout_p)
        self.ca_net = CA_NET()

    def forward(self, word_inputs, hidden):
        embedded = self.embed(word_inputs).transpose(0,1)
        output, hidden = self.gru(embedded, hidden)
        c_code, mu, logvar = self.ca_net(output)

        return c_code, hidden, mu, logvar

    def init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        return hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_dict, hidden_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.input_dict = input_dict
        self.attn = Attn(hidden_size, input_dict.max_len)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.palette_dim = 3

        self.gru = nn.GRUCell(self.hidden_size + self.palette_dim, hidden_size)

        self.out = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(hidden_size),
                        nn.Linear(hidden_size,self.palette_dim)
                   )

    def forward(self, last_palette, last_decoder_hidden, encoder_outputs, each_input_size, i):

        # Compute context vector.
        if i == 0:
            context = torch.mean(encoder_outputs, dim=1, keepdim=True)
        else:
            attn_weights = self.attn(last_decoder_hidden.squeeze(0), encoder_outputs, each_input_size)
            context = torch.bmm(attn_weights, encoder_outputs.transpose(0,1))

        # Compute gru output.
        gru_input = torch.cat((last_palette, context.squeeze(1)), 1)
        gru_hidden = self.gru(gru_input, last_decoder_hidden)

        # Generate palette color.
        palette = self.out(gru_hidden.squeeze(0))
        return palette, context.unsqueeze(0), gru_hidden, attn_weights


class Attn(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)
        self.attn_e = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_energy = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden, encoder_outputs, each_size):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        attn_energies = torch.zeros(seq_len,batch_size,1).cuda()

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        attn_energies = self.softmax(attn_energies) # (seq_len, batch_size, 1)
        return attn_energies.permute(1,2,0)         # (batch_size, 1, seq_len)

    def score(self, hidden, encoder_output):
        encoder_ = self.attn_e(encoder_output)  # encoder output (batch_size, hidden_size)
        hidden_ = self.attn_h(hidden)           # hidden (batch_size, hidden_size)
        energy = self.attn_energy(self.sigmoid(encoder_ + hidden_))

        return energy


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
