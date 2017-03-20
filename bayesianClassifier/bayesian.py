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
from random import shuffle, random
from random import shuffle

try:
    import cPickle as pickle
except:
    import pickle

tokenizer = RegexpTokenizer(r'\w+')
stopset = stopwords.words('english')
filteredWords = set(stopset + excludedWords)
# print "Getting type post entries"
types = list(
    set(post_dictionary[userID]["type"] for userID in post_dictionary.keys() if "type" in post_dictionary[userID]))
type_to_type_dict = {
    "ILE": ["ILE"],
    "LII": ["LII"],
    "SEI": ["SEI"],
    "ESE": ["ESE"],
    "SLE": ["SLE"],
    "LSI": ["LSI"],
    "EIE": ["EIE"],
    "IEI": ["IEI"],
    "LIE": ["LIE"],
    "ILI": ["ILI"],
    "SEE": ["SEE"],
    "ESI": ["ESI"],
    "LSE": ["LSE"],
    "SLI": ["SLI"],
    "EII": ["EII"],
    "IEE": ["IEE"]
}
type_to_temperament_dict = {
    'NT': [
        "ILE",
        "LII",
        "LIE",
        "ILI"
    ],
    'ST': [
        "SLE",
        "LSI",
        "LSE",
        "SLI"
    ],
    'NF': [
        "IEE",
        "EII",
        "EIE",
        "IEI"
    ],
    'SF': [
        "SEE",
        "ESI",
        "ESE",
        "SEI"
    ]
}

type_to_flow_dict = {
    "Static": [
        "ILE",
        "LII",
        "SLE",
        "LSI",
        "SEE",
        "ESI",
        "IEE",
        "EII"
    ],
    "Dynamic": [
        "ESE",
        "SEI",
        "EIE",
        "IEI",
        "LIE",
        "ILI",
        "LSE",
        "SLI"
    ]
}

type_to_aim_dict = {
    "Process": [
        "ILE",
        "SEI",
        "EIE",
        "LSI",
        "SEE",
        "ILI",
        "LSE",
        "EII"
    ],
    "Result": [
        "LII",
        "ESE",
        "SLE",
        "IEI",
        "LIE",
        "ESI",
        "IEE",
        "SLI"
    ]
}

type_to_aim_and_flow_dict = {
    "StaticProcess":
        [
            "ILE",
            "LSI",
            "SEE",
            "EII"
        ],  # CD
    "StaticResult":
        [
            "LII",
            "SLE",
            "ESI",
            "IEE"
        ],  # HP
    "DynamicProcess":
        [
            "SEI",
            "EIE",
            "ILI",
            "LSE"
        ],  # DA
    "DynamicResult":
        [
            "ESE",
            "IEI",
            "LIE",
            "SLI"
        ]  # VS
}

cur_dict = type_to_temperament_dict
cur_dict = {type_: cur for cur in cur_dict for type_ in
            cur_dict[cur]}
best_type = None
best_accuracy = 0.0
best_model = None
best_train_set, best_test_set = None, None
r = random()
for cur_type in tqdm(range(1)):
    # print cur_type
    train_set = []
    for userID in post_dictionary.keys():
        if "type" in post_dictionary[userID]:
            # if post_dictionary[userID]["username"].lower() == "myst":
            #     post_dictionary[userID]["type"] = cur_type
            for post in post_dictionary[userID]:
                if post not in EXCLUDED_KEYS:
                    post_words = list(tokenizer.tokenize(post_dictionary[userID][post]['content']))
                    train_set.append(({word: count for word, count in Counter(post_words).items() if
                                       count > 1 and len(word) > 2}, cur_dict[post_dictionary[userID]["type"]]))

                    # word.lower() for word in post_words if
                    # word.lower() not in filteredWords
    # print train_set[0]
    # print "Running trials"
    # for trial in tqdm(range(25)):
    # if t_i==0:
    shuffle(train_set, lambda: r)
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
        best_type = cur_type
        best_accuracy = accuracy
        best_model = classifier
        best_train_set = train_set
        best_test_set = test_set
    print classifier.show_most_informative_features()
while True:
    print classifier.classify(Counter(tokenizer.tokenize(raw_input())))

# for fname, obj in [("Model", best_model), ("TrainSet", best_train_set), ("TestSet", best_test_set),("Accuracy", best_accuracy)]:
#     with open('../output/bayesFrequency{0}.pickle'.format(fname), 'wb') as bFC:
#         data_string = pickle.dumps(obj)
#         bFC.write(data_string)
