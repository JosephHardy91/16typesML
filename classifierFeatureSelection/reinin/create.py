__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os
import pandas as pd
from reinin_words import *
import numpy as np
from tqdm import tqdm

from parse_data_in import post_dictionary

from sklearn.preprocessing import LabelEncoder

import cPickle as pickle


def check_membership(word, content):
    total = 0
    for sliver in content:
        if word in sliver:
            total += 1
    return total


def get_membership_count(content, class_words):
    return float(sum(map(lambda word: check_membership(word, content), class_words))) / len(content)


def get_membership_count_with_memberships(content, class_words):
    memberships = map(lambda word: check_membership(word, content), class_words)
    return float(sum(memberships)), memberships


def get_equal_member_word_proportions(content, word_group1, word_group2):
    # result of this needs to be normalized in df
    wg1_count, wg1_membership = get_membership_count_with_memberships(content, word_group1)
    wg2_count, wg2_membership = get_membership_count_with_memberships(content, word_group2)
    wg1_words_found = sum(map(lambda place: 1 if place > 0 else 0, wg1_membership))
    wg2_words_found = sum(map(lambda place: 1 if place > 0 else 0, wg2_membership))
    if wg1_count == 0 or wg2_count == 0:
        return 0.0
    if wg1_words_found < wg2_words_found:
        while wg1_words_found < len(wg2_membership):
            wg2_membership.remove(min(wg2_membership))

        wg2_count = sum(wg2_membership)

    elif wg1_words_found > wg2_words_found:
        while len(wg1_membership) > wg2_words_found:
                wg1_membership.remove(min(wg1_membership))

        wg1_count = sum(wg1_membership)
    wg1_count, wg2_count = map(float, (wg1_count, wg2_count))
    if wg1_count == 0 or wg2_count == 0:
        return 0.0
    return wg1_count/wg2_count


# Positivist/Negativist
def detect_pos_neg(post_content):
    return get_membership_count(post_content, neg_words)


# Carefree/Farsighted
def detect_cf_far(post_content):
    return get_equal_member_word_proportions(post_content, cf_words, far_words)


# Tactical/Strategic
def detect_tac_str(post_content):
    return get_equal_member_word_proportions(post_content, tac_words, str_words)


# Process/Result
def detect_pro_res(post_content):
    return get_equal_member_word_proportions(post_content, pro_words, res_words)


# Asking/Declaring 1
def detect_dec_ques_prop(post_content):
    periods = sum(1.0 for c in post_content if c == '.')
    questions = sum(1.0 for c in post_content if c == '?')

    return periods / questions if not (periods == 0 or questions == 0) else 0.0


# Asking/Declaring 2
def detect_length(post_content, max_length):
    return len(post_content) / max_length


labels = list(set([post_dictionary[userid]['type'] for userid in post_dictionary if 'type' in post_dictionary[userid]]))

le = LabelEncoder()

labels_transformed = le.fit_transform(labels)

pickle.dump(le, open('type_encoder.pickle', 'wb'), protocol=2)

# get clean post_list
max_len = 0
post_dict_by_type = defaultdict(list)
for userid in post_dictionary:
    if 'type' in post_dictionary[userid]:
        user_type = post_dictionary[userid]['type']
        for post_date in post_dictionary[userid]:
            if post_date not in ('username', 'type'):
                try:
                    post = post_dictionary[userid][post_date]['content'].lower()
                except:
                    raise IOError, "Could not parse %s" % post_date
                if len(post) > max_len:
                    max_len = len(post)
                post_dict_by_type[user_type].append(post)

# for lowercase post in post_dictionary, establish features and add to a dataframe
cols = ['PosNeg', 'CfFar', 'TacStr', 'ProRes', 'AskDec1', 'AskDec2', 'SType']
funcs = [detect_pos_neg, detect_cf_far, detect_tac_str, detect_pro_res, detect_dec_ques_prop,
         lambda x: detect_length(x, max_len)]
df = pd.DataFrame(columns=cols)
print 'Creating dataframe'
for type_ in post_dict_by_type.keys():
    print 'Parsing posts for %s' % type_
    for post in tqdm(post_dict_by_type[type_]):
        row = [func(post.split(" ")) for func in funcs]
        if not row == [0.0 for _ in range(len(funcs))]:
            df = df.append(dict(zip(cols, row+[le.transform([type_])[0]])),ignore_index=True)
    print '%d rows in df so far.' % df.shape[0]

print df.shape
print df.head(25)

pickle.dump(df, open('df.pickle', 'wb'), protocol=2)
