#python lstm.py
import math
import numpy as np

#inputs
x_in = np.random.rand(3)
h_in = np.random.rand(3)
c_in = np.random.rand(3)

#matrix dimensions
h_dim = len(h_in)
d_dim = len(x_in)

#matrices
w_f = np.random.rand(h_dim, d_dim)
u_f = np.random.rand(h_dim, h_dim)
b_f = 1

w_i = np.random.rand(h_dim, d_dim)
u_i = np.random.rand(h_dim, h_dim)
b_i = 1

w_o = np.random.rand(h_dim, d_dim)
u_o = np.random.rand(h_dim, h_dim)
b_o = 1

w_c = np.random.rand(h_dim, d_dim)
u_c = np.random.rand(h_dim, h_dim)
b_c = 1

#prepare hidden and current inputs for activation functions
f = np.dot(w_f, x_in) + np.dot(u_f, h_in) + b_f
i = np.dot(w_i, x_in) + np.dot(u_i, h_in) + b_i
o = np.dot(w_o, x_in) + np.dot(u_o, h_in) + b_o
_c = np.dot(w_c, x_in) + np.dot(u_c, h_in) + b_f

#activation functions
def sigmoid(inputs):
    return 1/(1 + np.exp(-inputs))

def tanh(inputs):
    return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

#1st layer outputs
f_out = sigmoid(f)
i_out = sigmoid(i)
o_out = sigmoid(o)
_c_out = tanh(_c)

#prepare 1st layer outputs for 2nd layer
i__c = i * _c
c_f = c_in * f

#memory state output
c_out = i__c + c_f

#hidden output
h_out = o_out * tanh(c_out)

print(h_out)
