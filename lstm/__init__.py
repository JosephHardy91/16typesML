__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

import numpy as np

np.random.seed(1337)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.backend import theano_backend

# from theano import tensor as T
#
# theano_backend.round = lambda x: T.round(x, mode='half_to_even')
from parse_data_in import post_dictionary, unique_type_list

import cPickle as pickle

import tqdm

max_words = 1000
batch_size = 32
nb_epoch = 500
all_scores = []
tokenizer = Tokenizer(nb_words=max_words)
tokenizer_fit = False
# type_to_train_for = 'LII'
for type_to_train_for in unique_type_list:
    print 'Wrangling data for %s model...' % type_to_train_for
    data = []
    # pbar = tqdm.tqdm()
    for user_id in post_dictionary:
        if 'type' in post_dictionary[user_id]:
            post_dict_keys = post_dictionary[user_id].keys()
            post_dict_keys.remove('type')
            post_dict_keys.remove('username')
            for post_date in post_dict_keys:
                data.append((post_dictionary[user_id][post_date]['content'],
                             True if post_dictionary[user_id]['type'] == type_to_train_for else False))
    X, y = map(np.array, zip(
        *data))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print X_train.shape[0], 'train sequences'
    # print X_test.shape[0], 'test sequences'

    if not tokenizer_fit:
        tokenizer.fit_on_texts(X_train)  # , mode='binary')
        pickle.dump(tokenizer, open('tokenizer.pickle', 'wb'), protocol=2)
        tokenizer_fit = True
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)  # , mode='binary')

    # print tokenizer.t

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    nb_classes = np.max(y_train) + 1
    # print(nb_classes, 'classes')

    # print('Vectorizing sequence data...')
    # tokenizer = Tokenizer(nb_words=max_words)
    # X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    # X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)

    # print X_train[0]

    # print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    # print('Y_train shape:', Y_train.shape)
    # print('Y_test shape:', Y_test.shape)

    # print Y_train[0]

    # pca = PCA(n_components=8)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)

    embedding_vector_length = 32
    hidden_layer_size = 500

    print('Building %s model...' % type_to_train_for)
    #model = Sequential()
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dense(hidden_layer_size, input_dim=hidden_layer_size))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # model.add(Embedding(X_train.shape[0], embedding_vector_length, input_length=max_words))
    # model.add(LSTM(15))
    # model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print model.summary()
    tqdm.tqdm(model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=1, batch_size=64, verbose=1))

    scores = model.evaluate(X_test, Y_test, verbose=0)
    all_scores.append((type_to_train_for, scores[1]))
    print 'Accuracy: %.3f%%' % (scores[1] * 100)
    pickle.dump(model, open('%s_model.pickle' % type_to_train_for, 'wb'), protocol=2)
    pickle.dump(le, open('%s_le.pickle' % type_to_train_for, 'wb'), protocol=2)
    model.save_weights('%s_model_weights.h5' % type_to_train_for)
for type_, score in all_scores:
    print "%s:%.2f" % (type_, score)
pickle.dump(all_scores, open('training_scores.pickle', 'wb'), protocol=2)
# model = Sequential()
# model.add(Dense(hidden_layer_size, input_shape=(8,)))
# model.add(Activation('relu'))
# model.add(Dense(hidden_layer_size, input_dim=hidden_layer_size))
# model.add(Activation('relu'))
# model.add(Dropout(0.05))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(X_train_pca, Y_train,
#                     nb_epoch=nb_epoch, batch_size=batch_size,
#                     verbose=1, validation_split=0.2)
# score = model.evaluate(X_test_pca, Y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# pickle.dump(X_train, 'X_train.pickle', protocol=2)
# pickle.dump(X_test, 'X_test.pickle', protocol=2)
# pickle.dump(le, 'labelencoded.pickle', protocol=2)
# pickle.dump(model, 'model.pickle', protocol=2)
