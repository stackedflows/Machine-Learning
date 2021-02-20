#python transformer.py
import numpy as np
import torch

#inputs (key, querys, value)
Q = np.random.rand(2, 6)
K = np.transpose(np.random.rand(2, 6))
V = np.random.rand(2, 3)

#layer 1
l_1 = K * np.transpose(Q)

#layer 2
l_2 = l_1 / np.sqrt(6)

#softmax activation function
softmax = torch.nn.Softmax(dim = 1)
l_2_tensor = torch.tensor(l_2)
l_3 = softmax(l_2_tensor)

#prepare V for last layer
V_tensor = torch.tensor(V)

#output layer
out = torch.mm(l_3, V_tensor)

print(out)
