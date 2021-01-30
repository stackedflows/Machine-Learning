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
        
        #to be constructed during the forward pass
        self.activations_matrixes = []
        
        #to be constructed during the backpropogation, and initialised next++
        self.weight_derivatives = []
        
        #to be constructed next
        self.weight_matrix = []
        
        weight_layer_0 = []
        derivatives_0 = []
        for i in range(self.inputs):
            partition_0 = []
            d_partitions_0 = []
            for ii in range(self.hidden_layers[0]):
                #to be used after testing: random_weight = 2 * random.random() - 1
                partition_0.append(1)
                d_partitions_0.append(0)
            weight_layer_0.append(partition_0)
            derivatives_0.append(d_partitions_0)
        self.weight_matrix.append(weight_layer_0)
        self.weight_derivatives.append(derivatives_0)
        
        #initialising hidden layers
        if len(self.hidden_layers) > 1:
            for i in range(1, len(self.hidden_layers)):
                weight_layer_i_hidden = []
                derivatives_layer_i_hidden = []
                for ii in range(self.hidden_layers[i - 1]):
                    partition_i = []
                    d_partitions_i = []
                    for iii in range(self.hidden_layers[i]):
                        #to be used after testing: random_weight = 2 * random.random() - 1
                        partition_i.append(1)
                        d_partitions_i.append(0)
                    weight_layer_i_hidden.append(partition_i)
                    derivatives_layer_i_hidden.append(d_partitions_i)
                self.weight_matrix.append(weight_layer_i_hidden)
                self.weight_derivatives.append(derivatives_layer_i_hidden)
                
        #initialising output weights layer
        weight_layer_last = []
        derivatives_layer_last = []
        for i in range(self.hidden_layers[len(self.hidden_layers) - 1]):
            partition_last = []
            d_partitions_last = []
            for ii in range(self.outputs):
                #to be used after testing: random_weight = 2 * random.random() - 1
                partition_last.append(1)
                d_partitions_last.append(0)
            weight_layer_last.append(partition_last)
            derivatives_layer_last.append(d_partitions_last)
        self.weight_matrix.append(weight_layer_last)
        self.weight_derivatives.append(derivatives_layer_last)
        
        print("weight matrix", self.weight_matrix)
        print("weight derivatives", self.weight_derivatives)
        print("activations", self.activations_matrixes)
        
        return
    
    #defining the dot product on two vectors
    def dot(self, set_0, set_1):
        product = []
        for element in range(len(set_0)):
            product.append(set_0[element] * set_1[element])
        return product
    
    #neuron activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    #method for passing new datum through the network, in the form [x,y,z ...]
    def forward_propogate_single(self, inputs):
        size = len(inputs)
        running_activations = [inputs]
        for i in range(0, len(self.weight_matrix)):
            weighted_layer = []
            for ii in range(len(self.weight_matrix[i])):
                weighted = self.weight_matrix[i][ii]
                for iii in range(len(weighted)):
                    weighted[iii] = weighted[iii] * running_activations[i][ii]
                weighted_layer.append(weighted)
            dot_prod = []
            for ii in range(len(weighted_layer) - 1):
                dot_prod = self.dot(weighted_layer[ii], weighted_layer[ii + 1])
            activations = dot_prod
            for ii in range(len(activations)):
                #activations[ii] = self.sigmoid(activations[ii])
                pass
            running_activations.append(activations)
        self.activations_matrixes.append(running_activations)
        print("forward pass activations", self.activations_matrixes)
        return 
    
    #propogates all input data
    def forward_propogate_all(self, input_training):
        size = len(input_training)
        for i in range(size):
            self.forward_propogate_single(input_training[i])
        return 
    
    #method for caluculationg errors
    def error_calculation(self, correct_outputs, outputs):
        errors = []
        for i in range(len(correct_outputs)):
            errors.append(correct_outputs[i] - outputs[i])
        return errors
            
    #defining the derivative of the sigmoid        
    def ddx_sigmoid(self, x):
        return x * (1 - x)
    
    #method for feeding errors backward in network
    def back_propogate_single(self, activations_matrix, weight_matrix, error):
        
        
        #for layer in reversed(range(0, len(weight_matrix))):
            #print(weight_matrix[layer])
            
        return
    
    def back_propogate_all(self, output_training):
        size = len(output_training)
        for i in range(size):
            self.back_propogate(output_training[i])
        return 
