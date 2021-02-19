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
sentence_index = {}

for sent, tags in data_train:
    for word in sent:
        if word not in tags:
            sentence_index[word] = len(sentence_index)

def to_tensor(sentence, sentence_index): 
    return torch.tensor([sentence_index[w] for w in sentence], dtype = torch.long)
#end preprocessing

#build model
class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

#size of input vector
EMBEDDING_DIM = 6
#size of hidden state
HIDDEN_DIM = 6

#initialise model
model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(sentence_index), len(tag_index))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

#train model
for epoch in range(200):
    for sentence, tags in data_train:
        #clear gradients
        model.zero_grad()
        
        sentence_in = to_tensor(sentence, sentence_index)
        targets = to_tensor(tags, tag_index)
        
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        
        #computes loss
        loss.backward()
        #applies optimisation
        optimizer.step()
        
        #observe model becomes sentient
        print(loss)
