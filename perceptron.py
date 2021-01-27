#python perceptron.py
import random
import math
import numpy as np

input_data = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]

weights = []

correct_outputs = [0, 1, 1, 0]

input_size = len(input_data[0])

training_iterations = 1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def ddx_sigmoid(x):
    return x * (1 - x)

def forward_pass(inputs, weights):
    outputs_sum = []
    neuron_count = len(inputs[0])
    for i in range(len(inputs)):
        new_output_sum = 0
        for ii in range(neuron_count):
            new_output_sum += inputs[i][ii] * weights[ii]
        outputs_sum.append(new_output_sum)
    for i in range(len(outputs_sum)):
        outputs_sum[i] = sigmoid(outputs_sum[i])
    return outputs_sum

def backpropogate(inputs, forward_outputs, correct_outputs, weights):
    errors = []
    for i in range(len(forward_outputs)):
        errors.append(forward_outputs[i] - correct_outputs[i])
    adjustments = []
    for i in range(len(errors)):
        adjustments.append(errors[i] * ddx_sigmoid(forward_outputs[i]))
    new_weights = []
    for i in range(len(weights)):
        new_weights.append(1 * adjustments[i])
    return new_weights

weights = 2 * np.random.random((3,1)) - 1

for i in range(10000):
    forward_outputs = forward_pass(input_data, weights)
    weights = backpropogate(input_data, forward_outputs, correct_outputs, weights)
    if i % 1000 == 0:
        print(forward_outputs)
        
