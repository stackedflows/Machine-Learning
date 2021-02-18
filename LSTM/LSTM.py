#python LSTM.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#training data
data_train = [
    ("The man ate the shoe".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Bill wants some cake".split(), ["NN", "V", "DET", "NN"]),
    ("Get to the Choppa".split(), ["I", "DET", "DET", "NN"])
]

#start pre-processing

#index each word type in dictionary
tag_index = {"DET": 0, "NN": 1, "V": 2, "I": 3}
#index each word 
index_ = {}

for sent, tags in data_train:
    for word in sent:
        if word not in tags:
            index_[word] = len(index_)

input_tensor = torch.tensor([index_[w] for w in data_train[0][0]])
#end preprocessing

#build model
class LSTM(nn.Module):
    def __init__(self, emb_dim, hidd_dim, index_size, tags_size):
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(index_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidd_dim)
        self.linear = nn.Linear(hidd_dim, tags_size)
        
    def forward(self, sentence):
        embs = word_embeddings(sentence)
        lstm_out, = self.lstm(embs.view(len(sentence), 1, -1))
        tags = self.linear(lstm_out.view(len(sentence), -1))
        scores = F.log_softmax(tags, dim = 1)
        return scores

#size of input vector
EMBEDDING_DIM = 6
#size of hidden state
HIDDEN_DIM = 6

model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(index_), len(tag_index))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

