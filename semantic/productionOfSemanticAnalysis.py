__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
# import pytreebank
from parse_data_in import post_dictionary
from textblob import TextBlob
from constants import EXCLUDED_KEYS

for userID in post_dictionary:
    for post in post_dictionary[userID]:
        if post in EXCLUDED_KEYS:
            continue
        post_sentiment = TextBlob(post_dictionary[userID][post]['content']).sentiment
        post_dictionary[userID][post].update(
            {'polarity': post_sentiment.polarity,
             'subjectivity': post_sentiment.subjectivity})

for userID in post_dictionary:
    avg_polarity, avg_subjectivity = 0.0, 0.0
    num_posts = len([p for p in post_dictionary[userID] if p not in EXCLUDED_KEYS])
    for post in post_dictionary[userID]:
        if post in EXCLUDED_KEYS:
            continue
        avg_polarity += post_dictionary[userID][post]['polarity']
        avg_subjectivity += post_dictionary[userID][post]['subjectivity']

    post_dictionary[userID].update(
        {'avg_polarity': avg_polarity / num_posts, 'avg_subjectivity': avg_subjectivity / num_posts})

# output
with open('output/semanticAnalysisResults.txt', 'w') as semanticAnalysisResults:
    for userID in post_dictionary:
        semanticAnalysisResults.write(",".join(
            map(str, [
                userID,
                post_dictionary[userID]['username'],
                post_dictionary[userID]['avg_polarity'],
                post_dictionary[userID]['avg_subjectivity']
            ]
                )) + "\n")
