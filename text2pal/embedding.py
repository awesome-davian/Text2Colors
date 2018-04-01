import pickle, os
import numpy as np

SOS_token = 0
EOS_token = 1

class Dictionary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        self.max_len = 0

    def index_elements(self, data):
        for element in data:
            self.max_len = len(data) if self.max_len < len(data) else self.max_len
            self.index_element(element)

    def index_element(self, element):
        if element not in self.word2index:
            self.word2index[element] = self.n_words
            self.word2count[element] = 1
            self.index2word[self.n_words] = element
            self.n_words += 1
        else:
            self.word2count[element] += 1


def prepare_data():
    input_dict = Dictionary()
    src_path = os.path.join('./data/hexcolor_vf/all_names.pkl')
    with open(src_path, 'rb') as f:
        text_data = pickle.load(f)
        f.close()

    print("Loading %s palette names..." % len(text_data))
    print("Making text dictionary...")

    for i in range(len(text_data)):
        input_dict.index_elements(text_data[i])

    return input_dict


def load_pretrained_embedding(dictionary, embed_file, embed_dim):
    if embed_file is None: return None

    pretrained_embed = {}
    with open(embed_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]
            if word == '<unk>':
                continue
            pretrained_embed[word] = entries
        f.close()

    vocab_size = len(dictionary) + 2
    W_emb = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for word, index in dictionary.items():
        if word in pretrained_embed:
            W_emb[index, :] = pretrained_embed[word]
            n += 1

    print ("%d/%d vocabs are initialized with GloVe embeddings." % (n, vocab_size))
    return W_emb