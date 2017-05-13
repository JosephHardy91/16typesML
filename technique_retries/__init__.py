__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from parse_data_in import post_dictionary
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import cPickle as pickle

# structure of post_dictionary
# user IDs
# post dates, username, type
# under post dates:
# content

# data input
posts, types = [], []
for userID in post_dictionary:
    for post in post_dictionary[userID]['posts']:
        posts.append(post_dictionary[userID]['posts'][post]['content'])
        types.append(post_dictionary[userID]['type'])

# preprocessing
tknzr = Tokenizer()
tknzr.fit_on_texts(posts)
tokenized_posts = sequence.pad_sequences(tknzr.texts_to_sequences(posts), maxlen=250)
X = tokenized_posts

le = LabelEncoder()
y = le.fit_transform(types)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# classifier training
print "Training"
gbc = GradientBoostingClassifier(verbose=1)
gbc.fit(X_train, y_train)

print "Scoring"
scoring = gbc.score(X_test, y_test)
print scoring

pickle.dump(gbc, open('./gradientBoostingClassifier.pickle', 'wb'), 2)
