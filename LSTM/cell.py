#python LSTM.py
import numpy as np
from numpy import random

#input vectors
input_t = []
hidden_t_0 = []
state_t_0 = [] #must be of len(input_t + hidden_t)

#learned behaviours
memory_matrix_throw = np.matrix() #must be of dimension len(input_t) x len(state_t_0)
bias_throw = 1
memory_matrix_gate_input = np.matrix()
bias_gate_input = 1
memory_matrix_state_input = np.matrix() 
bias_state_input = 1
memory_matrix_hidden_output = np.matrix() #must be of dimension len(state_t_0) x len(input_t)
bias_hidden_output = 1

#define useful functions
def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def tanh(x):
    x = np.array(x)
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def dot(x, y):
    return np.dot(x, y)

#concatenate input and previous hidden layer for activation processing
concat = hidden_t_0 + input_t

#run through activation function to decide which information to pass forward through the rest of the cell
throw_t = sigmoid(dot(memory_matrix_throw, concat) + bias_throw)

#run through 2 activation function to decide on new cell state
gate_input = sigmoid(dot(memory_matrix_gate_input, concat) + bias_gate_input)
state_input = tanh(dot(memory_matrix_state_input, concat) + bias_state_input)

#output cell state for this time step
output_state = dot(throw_t, state_t_0) + dot(gate_input, state_input)

#decide on hidden layer output for this time step
output_activation = sigmoid(dot(memory_matrix_hidden_output, concat) + bias_hidden_output)
hidden_output = dot(output_activation, tanh(output_state))

print(hidden_output)
