import json
import torch
from allennlp.models.reading_comprehension.util import get_best_span
import allennlp.nn.util as allenutil
from collections import defaultdict
import datasets.drop.constants as constants

trnfp = "/srv/local/data/nitishg/data/drop_old/date_subset/drop_dataset_train.json"
devfp = "/srv/local/data/nitishg/data/drop_old/date_subset/drop_dataset_dev.json"

with open(trnfp, 'r') as f:
    train_dateset = json.load(f)

with open(devfp, 'r') as f:
    dev_dateset = json.load(f)


def quesEvents(qstr):
    """ Returns (start, end) span tuples for event1 and event2 in the question """
    or_split = qstr.split(' or ')
    if len(or_split) != 2:
        return None

    tokens = qstr.split(' ')

    or_idx = tokens.index('or')
    # Last token is ? which we don't want to attend to
    event2 = tokens[or_idx + 1 : len(tokens) - 1]
    event2_span = (or_idx + 1, len(tokens) - 1)

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

    event1 = tokens[split_idx + 1: or_idx]
    event1_span = (split_idx + 1, or_idx)

    return event1_span, event2_span


def getFlippedQuestions(dataset):
    qa = []
    for passage_id, passage_info in dataset.items():
        for question_answer in passage_info[constants.qa_pairs]:
            question_tokenized_text = question_answer[constants.tokenized_question]
            original_question_text = question_answer[constants.original_question]
            question_charidxs = question_answer[constants.question_charidxs]
            event_spans = quesEvents(question_tokenized_text)
            if event_spans is None:
                continue

            event1_span, event2_span = event_spans

            tokens = question_tokenized_text.split(' ')
            pretext = tokens[0 : event1_span[0]]
            event2_text = tokens[event2_span[0] : event2_span[1]]
            mid_text = tokens[event1_span[1] : event2_span[0]]
            event1_text = tokens[event1_span[0] : event1_span[1]]
            end_text = tokens[event2_span[1]:]

            new_question_tokens = pretext + event2_text + mid_text + event1_text + end_text
            new_question_text = ' '.join(new_question_tokens)

            print(question_tokenized_text)
            print(new_question_text)
            print()

            # qa.append((question_text, span, first_or_last, events))
    return qa


if __name__=='__main__':
    tr_qa = getFlippedQuestions(train_dateset)
    dev_qa = getFlippedQuestions(dev_dateset)

    tr_size = len(tr_qa)
    dev_size = len(dev_qa)
    print(f"Number of tr: {tr_size}")
    print(f"Number of dev: {dev_size}")

