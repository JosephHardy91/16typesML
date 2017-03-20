__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

neg_words = map(str.lower, [
    "No",
    "Not",
    "None",
    "No one",
    "Nobody",
    "Nothing",
    "Neither",
    "Nowhere",
    "Never",
    "Hardly",
    "Scarcely",
    "Barely",
    "Doesn't",
    "Isn't",
    "Wasn't",
    "Shouldn't",
    "Wouldn't",
    "Couldn't",
    "Won't",
    "Can't",
    "Don't"
])

cf_words = map(str.lower, [
    'moment',
    'now',
    'hand',
])

far_words = map(str.lower, [
    'Experience',
    'Past',
    'Prepare',
    'Plan',
    'Advance',
    'Time'
])

tac_words = map(str.lower, [
    'Possibility',
    'Way',
    'Means',
    'Method',
    'Necessity',
    'Dream',
    'Interest',
    'Task'
])

str_words = map(str.lower, [
    'Goal',
    'Trajectory',
    'Aim',
    'System',
    'Project'
])

pro_words = map(str.lower, [
    'Process',
    'Grow',
    'Develop',
    'Progress'
])

res_words = map(str.lower, [
    'Beginning',
    'End',
    'Stage',
    'Interval',
    'Result',
    'Finish'
])
