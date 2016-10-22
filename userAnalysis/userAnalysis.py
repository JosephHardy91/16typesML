__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import matplotlib.pyplot as plt
from parse_data_in import post_dictionary
from constants import EXCLUDED_KEYS, TYPE_COLORS, TYPE_SHAPES
from textblob import TextBlob

n_post_dictionary = defaultdict(dict)
user_colors = defaultdict()
user_shapes = defaultdict()

for userID in post_dictionary:
    if "type" in post_dictionary[userID]:
        n_post_dictionary[userID] = post_dictionary[userID]
        user_colors[n_post_dictionary[userID]['username']] = TYPE_COLORS[n_post_dictionary[userID]['type']]
        user_shapes[n_post_dictionary[userID]['username']] = TYPE_SHAPES[n_post_dictionary[userID]['type']]

print len(post_dictionary)
print len(n_post_dictionary)

user_polarities = defaultdict(float)
user_subjectivities = defaultdict(float)
user_frequencies = defaultdict(int)

for userID in n_post_dictionary:
    # print n_post_dictionary[userID]['username']
    for post in n_post_dictionary[userID]:
        if post in EXCLUDED_KEYS:
            continue
        post_sentiment = TextBlob(n_post_dictionary[userID][post]['content']).sentiment
        user_polarities[n_post_dictionary[userID]['username']] += post_sentiment.polarity
        user_subjectivities[n_post_dictionary[userID]['username']] += post_sentiment.subjectivity
        user_frequencies[n_post_dictionary[userID]['username']] += 1

csize = 40

for user in user_frequencies:
    user_polarities[user] /= user_frequencies[user]
    user_subjectivities[user] /= user_frequencies[user]
    plt.scatter(user_subjectivities[user], user_polarities[user], c=user_colors[user], marker=user_shapes[user],
                s=csize)
    plt.annotate(user, xy=(user_subjectivities[user], user_polarities[user]))

plt.xlabel('subjectivity')
plt.ylabel('polarity')
plt.grid()
plt.show()
