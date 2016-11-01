__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from parse_data_in import post_dictionary
from constants import EXCLUDED_KEYS
from wordFrequency.wordFrequencyAnalysis import split_on_punctuation, excludedWords
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

try:
    import cPickle as pickle
except:
    import pickle
tokenizer = RegexpTokenizer(r'\w+')
stopset = stopwords.words('english')
filteredWords = set(stopset+excludedWords)
print "Getting type word counts"
type_word_counts = defaultdict(Counter)
for userID in tqdm(post_dictionary.keys()):
    if "type" in post_dictionary[userID]:
        for post in post_dictionary[userID]:
            if post not in EXCLUDED_KEYS:
                post_words = list(tokenizer.tokenize(post_dictionary[userID][post]['content']))
                type_word_counts[post_dictionary[userID]["type"]].update(
                    Counter([word.lower() for word in post_words if
                             word.lower() not in filteredWords])
                )
print "Writing data to pickle"
data_string = pickle.dumps(type_word_counts)
with open("../output/typewordcounts.pickle", 'wb') as out_file:
    out_file.write(data_string)
print "Done"
