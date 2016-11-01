__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from parse_data_in import post_dictionary, type_dictionary
from constants import EXCLUDED_KEYS
from wordFrequency.wordFrequencyAnalysis import split_on_punctuation, excludedWords
from tqdm import tqdm
import nltk
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from random import shuffle

try:
    import cPickle as pickle
except:
    import pickle
tokenizer = RegexpTokenizer(r'\w+')
stopset = stopwords.words('english')
filteredWords = set(stopset + excludedWords)
print "Getting type post entries"
train_set = []
for userID in tqdm(post_dictionary.keys()):
    if "type" in post_dictionary[userID]:
        for post in post_dictionary[userID]:
            if post not in EXCLUDED_KEYS:
                post_words = list(tokenizer.tokenize(post_dictionary[userID][post]['content']))
                train_set.append((Counter(post_words), post_dictionary[userID]["type"]))

                # word.lower() for word in post_words if
                # word.lower() not in filteredWords
# print train_set[0]
best_accuracy = 0.0
best_model = None
best_train_set, best_test_set = None, None
print "Running trials"
for trial in tqdm(range(25)):
    shuffle(train_set)
    training_set, test_set = train_set[:len(train_set) / 2], \
                             train_set[len(train_set) / 2:]

    # print "Training classifier"

    classifier = NaiveBayesClassifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, test_set)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = classifier
        best_train_set = train_set
        best_test_set = test_set

for fname, obj in [("Model", best_model), ("TrainSet", best_train_set), ("TestSet", best_test_set),
                   ("Accuracy", best_accuracy)]:
    with open('../output/bayesFrequency{0}.pickle'.format(fname), 'wb') as bFC:
        data_string = pickle.dumps(obj)
        bFC.write(data_string)
