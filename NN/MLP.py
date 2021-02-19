#non-linear functions can be represented with muli layering of perceptrons 
#the more layers/connection, the more fitted the data can become

import random, math, copy

class mlp:
    #ins must be integers, h_layers can be a 1 x n array of integers, out must be integers
    def __init__(self, ins = 2, h_layers = [3], outs = 2):
        #constants defining the layers of the network
        self.inputs = ins
        self.hidden_layers = h_layers
        self.outputs = outs
        
        #to be constructed during the forward pass
        self.activation_matrix = []
        
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
                random_weight = 2 * random.random() - 1
                partition_0.append(random_weight)
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
                        random_weight = 2 * random.random() - 1
                        partition_i.append(random_weight)
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
                random_weight = 2 * random.random() - 1
                partition_last.append(random_weight)
                d_partitions_last.append(0)
            weight_layer_last.append(partition_last)
            derivatives_layer_last.append(d_partitions_last)
        self.weight_matrix.append(weight_layer_last)
        self.weight_derivatives.append(derivatives_layer_last)
        
        #here we initialise the inverse weight matrix in preparation for back-propogation 
        self.inverse_weight_matrix = copy.copy(self.weight_matrix)
        self.inverse_weight_matrix.reverse()
        
        inverse = []
        for i in range(len(self.inverse_weight_matrix)):
            cell_size_previous = len((self.inverse_weight_matrix[i]))
            cell_size = len(self.inverse_weight_matrix[i][0])
            new_cell = []
            for ii in range(cell_size):
                partition_0 = []
                for iii in range(cell_size_previous):
                    partition_0.append(self.inverse_weight_matrix[i][iii][ii])
                new_cell.append(partition_0)
            inverse.append(new_cell)
        self.inverse_weight_matrix = inverse
        return
    
    #method to add position-wise elements in a tensor
    def add(self, set_0):
        sums = []
        for i in range(len(set_0[0])):
            added = 0
            for ii in range(len(set_0)):
                added += set_0[ii][i]
            sums.append(added)
        return sums
    
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
                weighted = copy.copy(self.weight_matrix[i][ii])
                for iii in range(len(weighted)):
                    weighted[iii] = weighted[iii] * running_activations[i][ii]
                weighted_layer.append(weighted)
            dot_prod = []
            for ii in range(len(weighted_layer)):
                dot_prod = self.add(weighted_layer)
            activations = dot_prod
            for ii in range(len(activations)):
                activations[ii] = self.sigmoid(activations[ii])
            running_activations.append(activations)
        self.activation_matrix = running_activations
        return 
            
    #defining the derivative of the sigmoid        
    def ddx_sigmoid(self, x):
        return x * (1 - x)
    
    #method for feeding errors backward in network
    def back_propogate_single(self, target):
        reverse_activations = self.activation_matrix
        reverse_activations.reverse()
        reversed_weight_matrix = copy.copy(self.inverse_weight_matrix)
        errors = []
        #initial errors
        for i in range(len(target)):
            errors.append(target[i] - reverse_activations[0][i])
        derivatives_matrix = []
        errors_matrix = []
        #backpropogation algorithm
        for i in range(len(reverse_activations) - 1):
            derivatives = []
            #errors * sigmoid derivative (activations)
            for ii in range(len(reverse_activations[i])):
                derivatives.append(errors[ii] * self.ddx_sigmoid(reverse_activations[i][ii]))
            dot_prod = []
            #propogate the errors into the next layer
            for ii in range(len(reversed_weight_matrix[i])):
                partition_1 = []
                for iii in range(len(reversed_weight_matrix[i][ii])):
                    partition_1.append(reversed_weight_matrix[i][ii][iii] * derivatives[ii])
                dot_prod.append(partition_1)
            derivatives_matrix.append(dot_prod)
            errors = self.add(dot_prod)
        return derivatives_matrix
    
    #updates weights based on back propgation single
    def gradient_descent_single(self, learning_rate, target):
        #adjust weights
        weights = copy.copy(self.inverse_weight_matrix)
        derivatives = self.back_propogate_single(target)
        for i in range(len(weights)):
            for ii in range(len(weights[i])):
                for iii in range(len(weights[i][ii])):
                    weights[i][ii][iii] += derivatives[i][ii][iii] * learning_rate
        weights_new = []
        #return matrix to non-inverse form
        for i in range(len(weights)):
            partition_0 = []
            this_cell_size = len(weights[i])
            cell_size = len(weights[i][0])
            weights_normal = []
            for j in range(cell_size):
                partition_1 = []
                for ii in range(this_cell_size):
                    partition_1.append(weights[i][ii][j])
                weights_normal.append(partition_1)
            weights_new.append(weights_normal)
        self.weight_matrix = weights_new
        self.weight_matrix.reverse()
        return 
    
    #trains network
    def train(self, inputs, targets, epochs, rate):
        for i in range(epochs):
            for ii in range(len(inputs)):
                self.forward_propogate_single(inputs[ii])
                self.gradient_descent_single(rate, targets[ii])
        return
