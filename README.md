# Single-Layer-Perceptron

relevant files : 

    perceptron.py

A implimentation of a single layer perceptron that I developed recently as an intorduction to python development

Recognises truth tables and makes predictions based on them

![2021-01-27 (2)](https://user-images.githubusercontent.com/73109076/106039816-7c9c9500-60d1-11eb-85c8-c2236c0455b4.png)

This is a sequence of steps from a recent learning session of 10000 iterations based on:

     inputs [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
 
     target [0, 1, 1, 0]
     

# Multi-Layer-Perceptron

relevent files:
    
    m_perceptron.py


It is a very generalised version of the SLP previously developed, capable of constructing as many hidden layers as needed and training the newtork on as many inputs and target values as wanted.

![2021-01-30 (2)](https://user-images.githubusercontent.com/73109076/106370704-7cf89280-6354-11eb-8925-6c8ca69f5941.png)

In this simple example, we see that It has learned to average the inputs, since the network cannot think of another way to optimise the error minimisation.

This is based on a 100 epoch training session and with a learning rate of 0.5.

# ~~ Application of ML To Unity: Various Frameworks~~

# Convolutional-Neural-Network

    relevant folder is MNIST

    relevent files: MNIST.py
                    Inference.py
                    
    
Builing a convolutional Neural Net In order to impliment MNIST classification, and exporting it to ONNX.

MNIST.py documented at https://github.com/pytorch/examples/tree/master/mnist

More details of MNIST at this link: http://yann.lecun.com/exdb/mnist/

The MNIST.py program will train a model, and after training, the Inference.py program will export it to a ONNX capable of being executed with barracuda.


    relevant folder is Unity


# Natural-Language-Processing

While unity are working on implementing GPT-3 type architectures (BERT) into their Barracuda package, here is a simple way to create a bot without such tools.

    relevant folder is Unity
