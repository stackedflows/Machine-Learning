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
        for i in range(self.inputs):
            partition_0 = []
            for ii in range(self.hidden_layers[0]):
                random_weight = 2 * random.random() - 1
                partition_0.append(1)
            weight_layer_0.append(partition_0)
        self.weight_matrix.append(weight_layer_0)
        
        #initialising hidden layers
        if len(self.hidden_layers) > 1:
            for i in range(1, len(self.hidden_layers)):
                weight_layer_i_hidden = []
                for ii in range(self.hidden_layers[i - 1]):
                    partition_i = []
                    for iii in range(self.hidden_layers[i]):
                        random_weight = 2 * random.random() - 1
                        partition_i.append(1)
                    weight_layer_i_hidden.append(partition_i)
                self.weight_matrix.append(weight_layer_i_hidden)
                
        #initialising output weights layer
        weight_layer_last = []
        for i in range(self.hidden_layers[len(self.hidden_layers) - 1]):
            partition_last = []
            for ii in range(self.outputs):
                random_weight = 2 * random.random() - 1
                partition_last.append(1)
            weight_layer_last.append(partition_last)
        self.weight_matrix.append(weight_layer_last)
        
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
    
    #method for passing new data through the network, in the form [x,y,z ...]
    def forward_propogate_single(self, inputs):
        activations = inputs
        for i in range(len(self.weight_matrix)):
            print("break")
            new_activations = []
            for ii in range(len(self.weight_matrix[i])):
                partitions_new_activations = []
                for iii in range(len(self.weight_matrix[i][ii])):
                    partitions_new_activations.append(activations[ii] * self.weight_matrix[i][ii][iii])
                new_activations.append(partitions_new_activations)
            dot_activations = []
            for ii in range(len(new_activations) - 1):
                dot_activations = self.dot(new_activations[ii], new_activations[ii + 1])
            activations = dot_activations
            for ii in range(len(activations)):
                activations[ii] = self.sigmoid(activations[ii])
            print(activations)
        
    def forward_propogate(self, input_training):
        size = len(input_training)
        forward_output = []
        for i in range(size):
            forward_output.append(self.forward_propogate_single(input_training[i]))
        print(forward_output)
        return forward_output
            
    #defining the derivative of the sigmoid        
    def ddx_sigmoid(self, x):
        return x * (1 - x)
    
    #method for adjusting network based on forward activations
    def back_propogate(self, correct_outputs):
        return 0
