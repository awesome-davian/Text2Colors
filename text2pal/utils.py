import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.autograd import Variable

class Embed(nn.Module):
    def __init__(self, vocab_size, embed_dim, W_emb, train_emb):
        super(Embed, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        if W_emb is not None:
            print ("Using pretrained W_emb...")
            self.embed.weight = nn.Parameter(W_emb)

        if train_emb == False:
            print ("Not training W_emb...")
            self.embed.requires_grad = False

    def forward(self, doc):
        doc = self.embed(doc)
        return doc

class MyModule(nn.Module):
    def __init__(self, hidden, dropout, batch_size):
        super(MyModule, self).__init__()
        self.linears_deep = nn.ModuleList([nn.Linear(hidden*i, hidden*(i+1)) for i in range(1,4)])
        self.linears_light = nn.ModuleList([nn.Linear(hidden*(5-i), hidden*(4-i)) for i in range(1,4)])
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden*2),
                            nn.BatchNorm1d(hidden*3),nn.BatchNorm1d(hidden*4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for i, l in enumerate(self.linears_deep):
            x = l(self.dropout(x))
            x = self.relu(x)
            x = self.batch_norm[i+1](x)
        for i, l in enumerate(self.linears_light):
            x = l(self.dropout(x))
            x = self.relu(x)
            x = self.batch_norm[2-i](x)

        return x


def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
