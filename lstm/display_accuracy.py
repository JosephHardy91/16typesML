__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import cPickle as pickle
from keras.preprocessing import sequence
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from parse_data_in import post_dictionary
from sklearn.model_selection import train_test_split
import random


def give_pred_symbol(a, b):
    return 'x' if a == b else 'o'


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

type_bounds = {
    'LII': (0, 3000),
    'ILI': (1000, 3000),
    'IEI': (2000, 3000),
    'EII': (3000, 3000),

    'ILE': (0, 2000),
    'LIE': (1000, 2000),
    'EIE': (2000, 2000),
    'IEE': (3000, 2000),

    'SEI': (0, 1000),
    'ESI': (1000, 1000),
    'LSI': (2000, 1000),
    'SLI': (3000, 1000),

    'ESE': (0, 0),
    'SEE': (1000, 0),
    'SLE': (2000, 0),
    'LSE': (3000, 0)
}


def give_coords(type_, length_of_post):
    cs = []
    for t, l in zip(type_, length_of_post):
        cs.append((l + type_bounds[t][0], l + type_bounds[t][1]))
    return cs


def count_nonzero(vec):
    return np.count_nonzero(vec)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=1000)
X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=1000)
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
marks = []
predictions = all_type_le.inverse_transform(forest.predict(tr_X))
pred_sym = np.vectorize(give_pred_symbol)
prediction_symbols = pred_sym(predictions, y_train)

# nonzero = np.vectorize(count_nonzero)
# nonzero_counts = nonzero(X_train)
nonzero_counts = np.count_nonzero(X_train, axis=1)
print nonzero_counts.shape[0]-np.count_nonzero(nonzero_counts)
# coord = np.vectorize(give_coords)
xycoords = give_coords(y_train, nonzero_counts)
print xycoords[0]

marks = zip(xycoords, prediction_symbols)

# for user in tqdm(filter(lambda user_:'type' in post_dictionary[user_].keys(),post_dictionary.keys())):
#     if 'type' in post_dictionary[user]:
#         type_ = post_dictionary[user]['type']
#         for post_date in [key for key in post_dictionary[user].keys() if key not in ['type', 'username']]:
#             post=post_dictionary[user][post_date]['content']
#             X = sequence.pad_sequences(tokenizer.texts_to_sequences(np.array([post])), maxlen=1000)
#             tokenized_X = np.zeros((1, 16))
#             for i, type_ in enumerate(unique_type_list):
#                 tokenized_X[:, i] = np.array(model_dictionary[type_]['model'].predict_proba(X, verbose=0))[:, 1]
#             prediction = forest.predict(tokenized_X)
#             if all_type_le.inverse_transform(prediction) == type_:
#                 prediction_result = 'x'
#             else:
#                 prediction_result = 'o'
#             length_of_post = np.count_nonzero(X)
#             marks.append(
#                 (length_of_post + type_bounds[type_][0], length_of_post + type_bounds[type_][1], prediction_result))
length = len(marks) / 5
for i, mark in tqdm(enumerate(marks)):
    if i % 10 == 0:
        # print mark
        (x, y), marker = mark
        jitter = random.uniform(-1.0,1.0) * 250
        plt.scatter(x + jitter, y - jitter, marker=marker)
for l_i in range(0, 4000, 1000):
    plt.axhline(l_i, 0, 4000)
    plt.axvline(l_i, 0, 4000)
for type_ in type_bounds:
    plt.text(type_bounds[type_][0] + 50, type_bounds[type_][1] + 950,type_)
plt.xlim((0,4000))
plt.ylim((0,4000))
plt.show()

print sum(predictions==y_train)/float(predictions.shape[0])

# while True:
#     post = raw_input('Enter text here> ')
#     print 'Performing prediction...',
#     X = sequence.pad_sequences(tokenizer.texts_to_sequences(np.array([post])), maxlen=1000)
#     tokenized_X = np.zeros((1, 16))
#     for i, type_ in tqdm(enumerate(unique_type_list)):
#         tokenized_X[:, i] = np.array(model_dictionary[type_]['model'].predict_proba(X, verbose=0))[:, 1]
#     prediction = forest.predict(tokenized_X)
#     print '\nPrediction for this post is: %s' % all_type_le.inverse_transform(prediction)
