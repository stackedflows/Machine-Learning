#python LSTM.py

import torch.nn

#dropout: randomly zeroes out elements of tensor input
#LSTM: layer of LSTM cells
#softmax: scales vecotr to [0,1]
#ReLU: max(0, x)
#CEL: log average difference
from torch.nn import Dropout, LSTM, Softmax, ReLU
from torch.nn import CrossEntropyLoss as CEL

import torchvision.datasets as datasets

mnist = datasets.MNIST(root = './data', train = True, download = True, transform = None)

model = torch.nn.Sequential(
)

#model optimiser
import torch.optim as optim
criterion = CEL()

#optimizer = optim.SGD()
