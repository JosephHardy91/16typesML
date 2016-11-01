__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pprint
try:
    import cPickle as pickle
except:
    import pickle

import typeWordPrevalence

print "Getting pickled object"
type_word_counts=None
with open("../output/typewordcounts.pickle") as tWCpickle:
    type_word_counts=pickle.loads(tWCpickle.read())

for type in type_word_counts:
    print type
    pprint.pprint(type_word_counts[type].most_common()[:10])