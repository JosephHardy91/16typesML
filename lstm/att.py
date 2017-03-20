__author__ = 'joe'

from collections import Counter, defaultdict
import sys, os

from keras.datasets import imdb

(Xtr,ytr),(xte,yte) = imdb.load_data(nb_words=5000)

print Xtr[0]

print ytr[0]