__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from vectorized import post_dictionary
from nltk.classify import NaiveBayesClassifier, accuracy
import numpy as np
from tqdm import tqdm


def unpack(string):
    return {}


# tfVec = TfidfVectorizer()
post_list = []
labels = []
for userID in post_dictionary:
    if 'type' in post_dictionary[userID]:
        for post in post_dictionary[userID]:
            if 'content' in post_dictionary[userID][post]:
                post_list.append(post_dictionary[userID][post]['content'])
                labels.append(post_dictionary[userID]['type'])
                # print len(post_list),len(labels)

post_list = np.array(post_list)
labels = np.array(labels)
del post_dictionary
X_train, X_test, y_train, y_test = map(list, train_test_split(post_list, labels, test_size=0.33))

print map(lambda t: len(t), [X_train, y_train, X_test, y_test])
# tfidf = TfidfVectorizer(stop_words='english')
# tf_train = tfidf.fit_transform(X_train)
# tf_test = tfidf.transform(X_test)

print 'Creating feature sets...\n'
# train_features = []
# for r in tqdm(range(tf_train.shape[0])):
#     train_features.append({c: tf_train[r, c] for c in range(tf_train.shape[1])})
# test_features = []
# for r in tqdm(range(tf_test.shape[0])):
#     test_features.append({c: tf_test[r, c] for c in range(tf_test.shape[1])})

cv = CountVectorizer(stop_words='english')
cv = cv.fit(X_train)
cvX_train = cv.transform(X_train)
cvX_test = cv.transform(X_test)
train_features = []
test_features = []
train_bar = tqdm(cvX_train,total=cvX_train.shape[0])
for i, train_row in enumerate(train_bar):
    train_val = train_row[0].toarray()
    train_features.append({j: train_val[0, j] for j in range(train_val.shape[1])})
    train_bar.update(1)
test_bar = tqdm(cvX_test,total=cvX_test.shape[0])
for i, test_row in enumerate(test_bar):
    test_val = test_row[0].toarray()
    test_features.append({j: test_val[0, j] for j in range(test_val.shape[1])})
    test_bar.update(1)

train_set = zip(train_features, y_train)
test_set = zip(test_features, y_test)
# train_set = np.hstack((X_train, y_train))
# test_set = np.hstack((X_test, y_test))

print 'Classifying...'
classifier = NaiveBayesClassifier.train(train_set)

print accuracy(classifier, test_set)

#
# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', #SGDClassifier(loss='hinge', penalty='l2',
#                               #             alpha=1e-3, n_iter=5, random_state=42)),
#                      ])
# for alpha in map(lambda p: 1.0 / 10 ** p, range(-5,10)):
#     text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
#                          ('fselec', LinearSVC(penalty='l2',dual=False,tol=alpha)),
#                          ('clf', LinearSVC()),
#                          ])
#
#     text_clf = text_clf.fit(X_train, y_train)
#     score = text_clf.score(X_test, y_test)
#     print alpha, score
