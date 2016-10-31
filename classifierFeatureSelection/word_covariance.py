__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from parse_data_in import post_dictionary
from constants import EXCLUDED_KEYS
from wordFrequency.wordFrequencyAnalysis import most_common_words
import numpy
from sklearn.covariance import EmpiricalCovariance
# from sklearn.linear_model import LinearRegression

# build regression model for each possible pair of words
# to model how word frequency changes with the prescense of either word

all_words = list(set(most_common_words))
# Get a set of points for each word pair counting the times that
# either of them appears in each post in which either of them appears once

word_pair_count = defaultdict(lambda: defaultdict(list))
for w_i, word in enumerate(all_words):
    for w_j, word2 in enumerate(all_words[w_i:]):
        for userID in post_dictionary:
            for post in post_dictionary[userID]:
                if post not in EXCLUDED_KEYS:
                    post_pair = [0.0, 0.0]
                    if word in post_dictionary[userID][post]['content']:
                        post_pair[0] = post_dictionary[userID][post]['content'].count(word)
                    if word2 in post_dictionary[userID][post]['content']:
                        post_pair[1] = post_dictionary[userID][post]['content'].count(word2)
                    word_pair_count[word][word2].append(post_pair)

covarianceModel = EmpiricalCovariance()
word_pair_covariances = defaultdict(lambda: defaultdict(float))
for word in word_pair_count:
    for word2 in word_pair_count[word]:
        xs, ys = zip(*word_pair_count[word][word2])
        word_pair_covariances[word][word2] = covarianceModel.fit(xs, ys)

with open('word_pair_covariances.csv', 'w') as wpc:
    wpc.write("Word1, Word2, Covariance\n")
    for word in word_pair_covariances:
        for word2 in word_pair_covariances[word]:
            wpc.write("{0},{1},{2}\n".format(word, word2, word_pair_covariances[word][word2]))

# type_word_covariances = defaultdict(lambda: defaultdict(float))
