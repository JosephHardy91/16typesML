__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

userSemantics = defaultdict(dict)

with open("output/semanticAnalysisResults.txt", "r") as semanticAnalysisResults:
    sARlines = semanticAnalysisResults.readlines()
    for line in sARlines:
        uid, name, polarity, subjectivity = line.split(",")
        if uid == "Creepy":
            continue
        lineDict = {'name': name, 'polarity': float(polarity), 'subjectivity': float(subjectivity)}
        userSemantics[int(uid)] = lineDict
