__author__ = 'joe'
import matplotlib.pyplot as plt

from semantic.semanticAnalysisReadin import userSemantics


def color_selector(cat1, cat2, cat1thres, cat2thres):
    if cat1 >= cat1thres:
        if cat2 >= cat2thres:
            return 'r'
        else:
            return 'orange'
    elif cat2 >= cat2thres:
        return 'g'
    else:
        return 'b'


for uid in userSemantics:
    s, p = [userSemantics[uid][k] for k in ['subjectivity', 'polarity']]
    plt.scatter(s, p, c=color_selector(s, p, 0.5, 0))

plt.show()
