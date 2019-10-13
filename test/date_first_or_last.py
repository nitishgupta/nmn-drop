import json
import torch
from allennlp.models.reading_comprehension.util import get_best_span
import allennlp.nn.util as allenutil
from collections import defaultdict
import datasets.drop.constants as constants

trnfp = "/srv/local/data/nitishg/data/drop_old/date_subset_augment/drop_dataset_train.json"
devfp = "/srv/local/data/nitishg/data/drop_old/date_subset_augment/drop_dataset_dev.json"

with open(trnfp, "r") as f:
    train_dateset = json.load(f)

with open(devfp, "r") as f:
    dev_dateset = json.load(f)


def gold_first_last(question):
    question_tokens = question.split(" ")
    lesser_tokens = ["first", "earlier", "forst", "firts"]
    greater_tokens = ["later", "last", "second"]

    for t in lesser_tokens:
        if t in question_tokens:
            return "first"

    for t in greater_tokens:
        if t in question_tokens:
            return "last"

    return "last"


def quesEvents(qstr):
    or_split = qstr.split(" or ")
    if len(or_split) != 2:
        return None

    tokens = qstr.split(" ")

    or_idx = tokens.index("or")
    # Last token is ? which we don't want to attend to
    event2 = tokens[or_idx + 1 : len(tokens) - 1]

    # Gets first index of the item
    try:
        comma_idx = tokens.index(",")
    except:
        comma_idx = 100000
    try:
        colon_idx = tokens.index(":")
    except:
        colon_idx = 100000

    try:
        hyphen_idx = tokens.index("-")
    except:
        hyphen_idx = 100000

    split_idx = min(comma_idx, colon_idx, hyphen_idx)

    if split_idx == 100000 or (or_idx - split_idx <= 1):
        # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
        if "first" in tokens:
            split_idx = tokens.index("first")
        elif "second" in tokens:
            split_idx = tokens.index("second")
        elif "last" in tokens:
            split_idx = tokens.index("last")
        elif "later" in tokens:
            split_idx = tokens.index("later")
        else:
            split_idx = -1

    assert split_idx != -1, f"{qstr} {split_idx} {or_idx}"

    event1 = tokens[split_idx + 1 : or_idx]

    return event1, event2


def getQuesAnsTuples(dataset):
    qa = []
    for passage_id, passage_info in dataset.items():
        for question_answer in passage_info[constants.qa_pairs]:
            question_text = question_answer[constants.tokenized_question]
            first_or_last = gold_first_last(question_text)
            answer_annotation = question_answer["answer"]
            span = answer_annotation["spans"][0]
            events = quesEvents(question_text)
            if events is None:
                continue
            qa.append((question_text, span, first_or_last, events))
    return qa


def first_or_last(qa_pairs):
    first_last_match = 0
    for qa_pair in qa_pairs:
        (question_text, span, first_or_last, events) = qa_pair
        ans_tokens = set(span.split(" "))
        event1, event2 = set(events[0]), set(events[1])
        ans_event = "first" if len(event1.intersection(ans_tokens)) > len(event2.intersection(ans_tokens)) else "last"
        if ans_event == first_or_last:
            # print(f"{ans_tokens}  ::  {event1}   {event2}")
            first_last_match += 1

    return first_last_match


if __name__ == "__main__":
    tr_qa = getQuesAnsTuples(train_dateset)
    dev_qa = getQuesAnsTuples(dev_dateset)

    tr_size = len(tr_qa)
    dev_size = len(dev_qa)
    print(f"Number of tr: {tr_size}")
    print(f"Number of dev: {dev_size}")

    print("First of last question type match event order in question: ")
    tr_match = first_or_last(tr_qa)
    dev_match = first_or_last(dev_qa)

    print(f"TR: {float(tr_match)/tr_size} ({tr_match})")
    print(f"DE: {float(dev_match)/dev_size} ({dev_match})")
