__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

import numpy as np

np.random.seed(1337)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

from parse_data_in import post_dictionary

import cPickle as pickle

max_words = 1000
batch_size = 32
nb_epoch = 500

print 'Wrangling data...'
data = []
for user_id in post_dictionary:
    if 'type' in post_dictionary[user_id]:
        post_dict_keys = post_dictionary[user_id].keys()
        post_dict_keys.remove('type')
        post_dict_keys.remove('username')
        for post_date in post_dict_keys:
            data.append((post_dictionary[user_id][post_date]['content'], post_dictionary[user_id]['type']))
X, y = map(np.array, zip(
    *data))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print X_train.shape[0], 'train sequences'
print X_test.shape[0], 'test sequences'

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

nb_classes = np.max(y_train) + 1
print(nb_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

hidden_layer_size = 1000

print('Building model...')
model = Sequential()
model.add(Dense(hidden_layer_size, input_shape=(8,)))
model.add(Activation('relu'))
model.add(Dense(hidden_layer_size, input_dim=hidden_layer_size))
model.add(Activation('relu'))
model.add(Dropout(0.05))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train_pca, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.2)
score = model.evaluate(X_test_pca, Y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

pickle.dump(X_train_pca,'X_train_pca.pickle',protocol=2)
pickle.dump(X_test_pca,'X_test_pca.pickle',protocol=2)
pickle.dump(le, 'labelencoded.pickle', protocol=2)
pickle.dump(model, 'model.pickle', protocol=2)
