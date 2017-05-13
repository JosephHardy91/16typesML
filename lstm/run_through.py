__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import cPickle as pickle
from keras.preprocessing import sequence
import numpy as np
from tqdm import tqdm

print 'Loading tokenizer...',
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
print 'Loaded.\nLoading forest...',
forest = pickle.load(open('forest_model.pickle', 'rb'))
print 'Loaded.\nLoading label encoder...',
all_type_le = pickle.load(open('all_type_le.pickle', 'rb'))
print 'Loaded.\nLoading type list...',
unique_type_list = pickle.load(open('u_type_list.pickle', 'rb'))
print 'Loaded.\nLoading individual models and label encoders...',
model_dictionary = defaultdict(dict)
for type_ in unique_type_list:
    for model_part in ('model', 'le'):
        model_dictionary[type_][model_part] = pickle.load(open('%s_%s.pickle' % (type_, model_part), 'rb'))
    model_dictionary[type_]['model'].load_weights('%s_model_weights.h5' % type_)
print 'Loaded.'

while True:
    post = raw_input('Enter text here> ')
    print 'Performing prediction...',
    X = sequence.pad_sequences(tokenizer.texts_to_sequences(np.array([post])), maxlen=1000)
    tokenized_X = np.zeros((1, 16))
    for i, type_ in tqdm(enumerate(unique_type_list)):
        tokenized_X[:, i] = np.array(model_dictionary[type_]['model'].predict_proba(X, verbose=0))[:, 1]
    prediction = forest.predict(tokenized_X)
    print '\nPrediction for this post is: %s' % all_type_le.inverse_transform(prediction)
