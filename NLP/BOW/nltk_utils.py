#python nltk_utils.py

import nltk
import numpy as np

#nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentance):
    return nltk.word_tokenize(sentance)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    
    bag = np.zeros(len(all_words), dtype = np.float32)
    
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag
    
