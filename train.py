import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize,sent_tokenize

#famous algorithm : to learn!
stemmer = LancasterStemmer()

#dependencies
import numpy
import tflearn
from tensorflow.python.framework import ops
import random
import json
import tf2onnx

#parse json
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

#natural language processing bit
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        #tokenise
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

#stemming: creates list of fundamental words
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]

#sorting
words = sorted(list(set(words)))
labels = sorted(labels)

#bag of words: corresponds words to strings, and whether they exist by frequency
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

#training the neural net
training = numpy.array(training)
output = numpy.array(output)

ops.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch = 2000, batch_size = 8, show_metric = True)
model.save("model.tflearn")
