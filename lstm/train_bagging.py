__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import cPickle as pickle

import theano

from keras.models import Sequential
from keras.preprocessing import sequence
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.backend import theano_backend
from sklearn.model_selection import train_test_split

import numpy as np
from parse_data_in import post_dictionary, unique_type_list

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder


# get_last_layer_output = K.function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
model_dictionary = defaultdict(dict)
for type_ in unique_type_list:
    for model_part in ('model', 'le'):
        model_dictionary[type_][model_part] = pickle.load(open('%s_%s.pickle' % (type_, model_part), 'rb'))
    model_dictionary[type_]['model'].load_weights('%s_model_weights.h5' % type_)
    # model_dictionary[type_]['model'].verbose = 0
data = []
# pbar = tqdm.tqdm()
print 'Wrangling data'
for user_id in tqdm(post_dictionary.keys()):
    if 'type' in post_dictionary[user_id]:
        post_dict_keys = post_dictionary[user_id].keys()
        post_dict_keys.remove('type')
        post_dict_keys.remove('username')
        for post_date in post_dict_keys:
            data.append((post_dictionary[user_id][post_date]['content'],
                         post_dictionary[user_id]['type']))
X, y = map(np.array, zip(
    *data))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=1000)
X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=1000)

# for type_ in unique_type_list:
#     print type_
#     y_test_type = np.zeros(y_test.shape[0])
#     #print np.unique(y_test_type)
#     #y_test[y_test != type_] = 0
#     y_test_type[y_test == type_] = 1
#     print np.unique(y_test)
#     print model_dictionary[type_]['model'].evaluate(X_test, np_utils.to_categorical(y_test_type, 2), verbose=0)[1]

tr_X = np.zeros((X_train.shape[0], 16))
te_X = np.zeros((X_test.shape[0], 16))
# tr_Y = np.zeros(())
correct_predictions = 0.0
print 'Getting predictions for each type'
for i, type_ in tqdm(enumerate(unique_type_list)):
    # print model_dictionary[type_]
    # y_train_type = np.zeros(y_train.shape[0])
    # y_train_type[y_train==type_]=1
    tr_probs = np.array(model_dictionary[type_]['model'].predict_proba(X_train, verbose=0))
    # proba_score = np.sum(np.argmax(tr_probs, axis=1) == y_train_type) / float(tr_probs.shape[0])
    # print proba_score, \
    # model_dictionary[type_]['model'].evaluate(X_train, np_utils.to_categorical(y_train_type, 2), verbose=0)[1]
    tr_X[:, i] = tr_probs[:, 1]
    te_probs = np.array(model_dictionary[type_]['model'].predict_proba(X_test, verbose=0))
    te_X[:, i] = te_probs[:, 1]

print 'Getting overall predictions'
print ' '.join(unique_type_list)
all_type_le = LabelEncoder()
y_tr = all_type_le.fit_transform(y_train)
y_te = all_type_le.transform(y_test)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt

# scores = []
# for n_estimators_exp in tqdm(range(1, 5)):
# n_estimators = 10 * 10 ** n_estimators_exp
n_forests = 1
forest_data = np.zeros((tr_X.shape[0], 1, 16))
# forest_data[:, n_forests] = y_tr[:,np.newaxis]
# for forest_num in tqdm(range(n_forests)):
#     if forest_num < n_forests / 2.0:
#         criteron = 'gini'
#     else:
#         criteron = 'entropy'
acc = []
for n_est in range(5, 100, 5):
    forest = ExtraTreesClassifier(criterion='gini', n_estimators=10, random_state=1, n_jobs=-1)

    forest.fit(tr_X, y_tr)
    forest_data[:, 0, :] = forest.predict_proba(tr_X)
    acc.append((n_est, np.sum(forest.predict(tr_X) == y_tr) / float(tr_X.shape[0])))  # 90.36%

steps, results = zip(*acc)
plt.plot(steps, results)
plt.show()

for step, result in acc:
    if result >= 0.90:
        forest = ExtraTreesClassifier(criterion='gini', n_estimators=step, random_state=1, n_jobs=-1)

        forest.fit(tr_X, y_tr)
        forest_data[:, 0, :] = forest.predict_proba(tr_X)
        print np.sum(forest.predict(tr_X) == y_tr) / float(tr_X.shape[0])  # 90.36%
        break

pickle.dump(forest_data, open('forested_data.pickle', 'wb'), protocol=2)
pickle.dump(forest, open('forest_model.pickle', 'wb'), protocol=2)
pickle.dump(y_tr, open('y_train.pickle', 'wb'), protocol=2)
pickle.dump(all_type_le, open('all_type_le.pickle', 'wb'), protocol=2)
pickle.dump(unique_type_list, open('u_type_list.pickle', 'wb'), protocol=2)

# plt.plot(range(1, 5), scores)
# plt.show()

# Y_train = np_utils.to_categorical(y_tr, 16)
# Y_test = np_utils.to_categorical(y_te, 16)
#
# hidden_layer_size = 1000
# model = Sequential()
# model.add(Dense(hidden_layer_size, input_shape=(tr_X.shape[1],)))
# model.add(Activation('relu'))
# model.add(Dense(hidden_layer_size, input_dim=hidden_layer_size))
# model.add(Activation('relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(16))
# model.add(Activation('softmax'))
# # model.add(Embedding(X_train.shape[0], embedding_vector_length, input_length=max_words))
# # model.add(LSTM(15))
# # model.add(Dense(nb_classes, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # print model.summary()
# model.fit(tr_X, Y_train, validation_data=(te_X, Y_test), nb_epoch=100, batch_size=64, verbose=1)
#
# scores = model.evaluate(te_X, Y_test, verbose=0)
# print 'Accuracy: %.3f%%' % (scores[1] * 100)  # from sklearn.linear_model import LogisticRegression
#
# log_reg = LogisticRegression()
# log_reg.fit(tr_X, y_train)
# train_accuracy = log_reg.score(tr_X, y_train)
# test_accuracy = log_reg.score(te_X, y_test)
#
# print train_accuracy, test_accuracy

# for idx in tqdm(xrange(tr_X.shape[0])):
#     #     new_X[:, 0] = np.array(
#     #         [model_dictionary[type_]['model'].predict_proba(
#     #             X_train[idx]) for type_ in unique_type_list])
#     pred = unique_type_list[np.argmax(tr_X[idx, :], axis=0)]
#     print y_train[idx], pred, zip(unique_type_list, tr_X[idx, :])
#     if pred == y_train[idx]:
#         correct_predictions += 1.0
#
# print correct_predictions
# print correct_predictions / X_train.shape[0]
