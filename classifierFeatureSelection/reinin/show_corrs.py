__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
# 240 corrs - 1 per 2-feature combination per type (15 2-feature combinations, 16 types)
import cPickle as pickle
import numpy as np
import theano

np.random.seed(17)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD,Nadam
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

df = pickle.load(open('df.pickle', 'rb'))
le = pickle.load(open('type_encoder.pickle'))
data = df.as_matrix()
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

y_train_ohe = np_utils.to_categorical(y_train, 16)

nb_epoch = 50000
batch_size = 32

layer_size = 5000

model = Sequential()  # this is like a pipeline, where one thing happens after another
# adding layers in the neural network (model)
model.add(Dense(input_dim=X_train.shape[1],
                output_dim=layer_size,
                init='uniform',
                activation='tanh'))

model.add(Dense(input_dim=layer_size,
                output_dim=layer_size,
                init='uniform',
                activation='softplus'))

#model.add(Dropout(0.05))

model.add(Dense(input_dim=layer_size,
                output_dim=16,
                init='uniform',
                activation='softmax'))

#sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)
sgd = Nadam()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train_ohe, nb_epoch=nb_epoch, batch_size=3000, verbose=1, validation_split=0.1)

y_train_pred = model.predict_classes(X_train, verbose=0)
print 'First 3 predictions: ', y_train_pred[:3], y_train[:3] == y_train_pred[:3]

train_acc = np.sum(y_train == y_train_pred, axis=0) / float(X_train.shape[0])
print 'Training accuracy: %.2f%%' % (train_acc * 100.0)

y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(y_test == y_test_pred, axis=0) / float(X_test.shape[0])
print 'Test accuracy: %.2f%%' % (test_acc * 100.0)

pickle.dump(le, open('labelencoded.pickle', 'wb'), protocol=2)
pickle.dump(model, open('model.pickle', 'wb'), protocol=2)
