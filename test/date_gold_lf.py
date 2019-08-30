import json
import torch
from allennlp.models.reading_comprehension.util import get_best_span
import allennlp.nn.util as allenutil
from collections import defaultdict

trnfp = "/srv/local/data/nitishg/data/drop_old/date_subset/train_ques.txt"
devfp = "/srv/local/data/nitishg/data/drop_old/date_subset/dev_ques.txt"

with open(trnfp, 'r') as f:
    train_questions = f.read().strip().split('\n')

with open(devfp, 'r') as f:
    dev_questions = f.read().strip().split('\n')


print(f"Num of train question: {len(train_questions)}")
print(f"Num of dev question: {len(dev_questions)}")


date_lesser = "date_lesser"
date_greater = "date_greater"

def questionAttns(qstr):
    tokens = qstr.split(' ')


    lesser_tokens = ['first', 'earlier', 'forst', 'firts']
    greater_tokens = ['later', 'last', 'second']

    func = None

    for t in lesser_tokens:
        if t in tokens:
            return date_lesser

    for t in greater_tokens:
        if t in tokens:
            return date_greater

    print(qstr)
    return None


def quesAttnsDataset(questions):
    not_parsed = 0
    func_counter = defaultdict(int)

    for q in questions:

        func = questionAttns(q)
        func_counter[func] += 1

        print(f"{func} : {q}")

        if not func:
            not_parsed += 1

    print(func_counter)
    print(f"Not parsed: {not_parsed}")



if __name__=='__main__':
    print("\nTrain questions")
    quesAttnsDataset(train_questions)

    print('\nDev questions')
    quesAttnsDataset(dev_questions)

