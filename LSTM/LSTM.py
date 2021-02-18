#python LSTM.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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

input_tensor = torch.tensor([index_[w] for w in data_train[0][0]], index_)
#end preprocessing

#size of input vector
EMBEDDING_DIM = 3
#size of hidden state
HIDDEN_DIM = 3

#build model
embedding = nn.Embedding(len(index_), EMBEDDING_DIM)
emb = embedding(input_tensor)
lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
linear = nn.Linear(HIDDEN_DIM, len(index_))


model = F.log_softmax(model, dim = 1)
