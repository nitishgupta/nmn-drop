import json
import torch
from allennlp.models.reading_comprehension.util import get_best_span
import allennlp.nn.util as allenutil
from collections import defaultdict
import datasets.drop.constants as constants

trnfp = "/srv/local/data/nitishg/data/drop_old/date_subset/train_ques.txt"
devfp = "/srv/local/data/nitishg/data/drop_old/date_subset/dev_ques.txt"

with open(trnfp, 'r') as f:
    train_questions = f.read().strip().split('\n')

with open(devfp, 'r') as f:
    dev_questions = f.read().strip().split('\n')

def questionAttns(qstr):
    or_split = qstr.split(' or ')
    if len(or_split) != 2:
        return (0, 0)

    tokens = qstr.split(' ')
    attn_1 = [0 for _ in range(len(tokens))]
    attn_2 = [0 for _ in range(len(tokens))]

    or_idx = tokens.index('or')
    # Last token is ? which we don't want to attend to
    attn_2[or_idx + 1 : len(tokens) - 1] = [1] * (len(tokens) - or_idx - 2)
    assert sum(attn_2) > 0
    assert len(tokens) == len(attn_2)

    # Gets first index of the item
    try:
        comma_idx = tokens.index(',')
    except:
        comma_idx = 100000
    try:
        colon_idx = tokens.index(':')
    except:
        colon_idx = 100000

    try:
        hyphen_idx = tokens.index('-')
    except:
        hyphen_idx = 100000

    split_idx = min(comma_idx, colon_idx, hyphen_idx)

    if split_idx == 100000 or (or_idx - split_idx <= 1):
        # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
        if 'first' in tokens:
            split_idx = tokens.index('first')
        elif 'second' in tokens:
            split_idx = tokens.index('second')
        elif 'last' in tokens:
            split_idx = tokens.index('last')
        elif 'later' in tokens:
            split_idx = tokens.index('later')
        else:
            split_idx = -1

    assert split_idx != -1, f"{qstr} {split_idx} {or_idx}"

    attn_1[split_idx + 1: or_idx] = [1] * (or_idx - split_idx - 1)

    assert sum(attn_1) > 0, f"{qstr} {split_idx} {or_idx}"
    assert len(tokens) == len(attn_1)

    # print([(x,y,z) for x,y,z in zip(tokens, attn_1, attn_2)])

    return attn_1, attn_2



def quesAttnsDataset(questions):
    or_split_count = defaultdict(int)

    for q in questions:
        (a1, a2) = questionAttns(q)

    print("all done")



if __name__=='__main__':
    quesAttnsDataset(train_questions)
    quesAttnsDataset(dev_questions)

