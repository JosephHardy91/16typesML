__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import matplotlib.pyplot as plt
from parse_data_in import post_dictionary
from constants import EXCLUDED_KEYS
from textblob import TextBlob

n_post_dictionary = defaultdict(dict)

for userID in post_dictionary:
    if "type" in post_dictionary[userID]:
        n_post_dictionary[userID] = post_dictionary[userID]

print len(post_dictionary)
print len(n_post_dictionary)

type_polarities = defaultdict(float)
type_subjectivities = defaultdict(float)
type_frequencies = defaultdict(int)

for userID in n_post_dictionary:
    #print n_post_dictionary[userID]['username']
    for post in n_post_dictionary[userID]:
        if post in EXCLUDED_KEYS:
            continue
        post_sentiment = TextBlob(n_post_dictionary[userID][post]['content']).sentiment
        type_polarities[n_post_dictionary[userID]['type']] += post_sentiment.polarity
        type_subjectivities[n_post_dictionary[userID]['type']] += post_sentiment.subjectivity
        type_frequencies[n_post_dictionary[userID]['type']] += 1

for type in type_frequencies:
    type_polarities[type] /= type_frequencies[type]
    type_subjectivities[type] /= type_frequencies[type]

    plt.scatter(type_subjectivities[type], type_polarities[type])
    plt.annotate(type, xy=(type_subjectivities[type], type_polarities[type]))

plt.xlabel('subjectivity')
plt.ylabel('polarity')
plt.grid()
plt.show()
