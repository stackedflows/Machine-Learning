#python m_perceptron.py

#non-linear functions can be represented with muli layering of perceptrons 
#the more layers/connection, the more fitted the data can become

import random
import math

class mlp:
    def __init__(self, ins = 2, h_layers = [3, 2, 3], outs = 2):
        #constants defining the layers of the network
        self.inputs = ins
        self.hidden_layers = h_layers
        self.outputs = outs

        self.weight_matrix = []
        
        #initialising inputs weight layers
        weight_layer_0 = []
        for i in range(self.inputs * self.hidden_layers[0]):
            random_weight = 2 * random.random() - 1
            weight_layer_0.append(random_weight)
        self.weight_matrix.append(weight_layer_0)
        
        #initialising hidden layers 
        if len(self.hidden_layers) > 1:
            for i in range(1, len(self.hidden_layers)):
                weight_layer_i_hidden = []
                for ii in range(self.hidden_layers[i - 1] * self.hidden_layers[i]):
                    random_weight = 2 * random.random() - 1
                    weight_layer_i_hidden.append(random_weight)
                self.weight_matrix.append(weight_layer_i_hidden)
                
        #initialising output weights layer
        weight_layer_last = []
        for i in range(self.outputs * self.hidden_layers[len(self.hidden_layers) - 1]):
            random_weight = 2 * random.random() - 1
            weight_layer_last.append(random_weight)
        self.weight_matrix.append(weight_layer_last)
        
        return
    
    def dot(self, set_0, set_1):
        product = []
        for element in range(len(set_0)):
            product.append(set_0[element] * set_1[element])
        return product
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def forward_propogate(self, inputs):
        
        #check to see if length is matching for all training data
        check = 0
        for i in range(len(inputs)):
            if len(inputs[i]) != self.inputs:
                check += 1
        assert check == 0
        
        activations = inputs
        
        #propogate inputs ## debug here ##
        for i in range(len(self.weight_matrix)):
            activations = self.dot(activations, self.weight_matrix[i])
            
            
    def ddx_sigmoid(self, x):
        return x * (1 - x)
    
    def back_propogate(self):
        return 0
