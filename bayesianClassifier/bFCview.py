__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import nltk
from nltk import NaiveBayesClassifier
#import bayesian

try:
    import cPickle as pickle
except:
    import pickle
bFCfile = open('../output/bayesFrequencyClassifier.pickle', 'rb').read()
bFC = pickle.loads(bFCfile)

print nltk.classify.accuracy(bFC,)

print bFC.most_informative_features(5)
