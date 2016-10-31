__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
from parse_data_in import post_dictionary
from constants import EXCLUDED_KEYS


#   post_dictionary[userID][dateTime] = {'content': content, 'username': username}

def clean_word(word):
    junk = ["\"", ")", "(", "'", "[", "]", "{", "}", "=", "8", "%", "#", "@"]
    for junk_part in junk:
        word = word.replace(junk_part, '')
    return word.lower()


def split_on_punctuation(word_list):
    new_word_list = []
    splitters = ["/", ".", "-", "?", ":", "!", ";", ",", "\\", "&"]
    for word in word_list:
        word_parts = word.split()
        for splitter in splitters:
            new_word_parts = []
            for part in word_parts:
                split_word_parts = filter(lambda w: w not in excludedWords,
                                          [wpart for wpart in map(clean_word, part.split(splitter)) if len(wpart) > 0])
                new_word_parts.extend(split_word_parts)
            word_parts = new_word_parts
        new_word_list.extend(word_parts)
    return new_word_list


excludedWords = [y.strip().lower() for y in open('../data/wordsToExclude.txt', 'r').readlines()] + [str(num) for num in
                                                                                                 range(0, 100)]

most_common_words = Counter()
for userID in post_dictionary:
    for post in post_dictionary[userID]:
        if post not in EXCLUDED_KEYS:
            words = most_common_words.update(
                Counter(split_on_punctuation(post_dictionary[userID][post]['content'].split())))

print len(most_common_words)

with open('../output/wordFrequencies.csv', 'w') as wordFrequencies:
    for word in sorted(most_common_words.keys(), key=lambda x: most_common_words[x], reverse=True):
        wordFrequencies.write(word + "," + str(most_common_words[word]) + "\n")
