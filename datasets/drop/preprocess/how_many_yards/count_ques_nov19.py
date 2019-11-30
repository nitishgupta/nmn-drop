from typing import List
import json
from nltk.corpus import stopwords
import os
import re
from collections import defaultdict
import datasets.drop.constants as constants
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["'s", ","])

FIRST = "first"
SECOND = "second"

COUNT_TRIGRAMS = [
    "how many field goals did",
    "how many field goals were",
    "how many interceptions did",
    "how many passes",
    "how many rushing",
    "how many touchdown passes did",
    "how many touchdowns did the",
    "how many touchdowns were scored",
]


COUNT_FIND_QUESTION_REGEX_PATTERNS = [
    "how many touchdowns did \w+ \w+ \?",
    "how many touchdowns did \w+ \w+ \w+ \?",
    "how many rushing touchdowns did \w+ \w+ \?",
    "how many rushing touchdowns did \w+ \w+ \w+ \?",
    "how many touchdown passes did \w+ \w+ \?",
    "how many touchdown passes did \w+ \w+ \w+ \?",
    "how many passes did \w+ \w+ \?",
    "how many passes did \w+ \w+ \w+ \?",
    "how many field goals did \w+ \w+ \?",
    "how many field goals did \w+ \w+ \w+ \?",
]

COUNT_FILTER_FIND_QUESTION_REGEX_PATTERNS = [
    "how many touchdowns did \w+ \w+ during the \w+ \w+ \?",
    "how many touchdowns did \w+ \w+ \w+ during the \w+ \w+ \?",
    "how many rushing touchdowns did \w+ \w+ during the \w+ \w+ \?",
    "how many rushing touchdowns did \w+ \w+ \w+ during the \w+ \w+ \?",
    "how many touchdown passes did \w+ \w+ during the \w+ \w+ \?",
    "how many touchdown passes did \w+ \w+ \w+ during the \w+ \w+ \?",
    "how many passes did \w+ \w+ during the \w+ \w+ \?",
    "how many passes did \w+ \w+ \w+ during the \w+ \w+ \?",
    "how many field goals did \w+ \w+ during the \w+ \w+ \?",
    "how many field goals did \w+ \w+ \w+ during the \w+ \w+ \?",
    "how many touchdowns did \w+ \w+ in the \w+ \w+ \?",
    "how many touchdowns did \w+ \w+ \w+ in the \w+ \w+ \?",
    "how many rushing touchdowns did \w+ \w+ in the \w+ \w+ \?",
    "how many rushing touchdowns did \w+ \w+ \w+ in the \w+ \w+ \?",
    "how many touchdown passes did \w+ \w+ in the \w+ \w+ \?",
    "how many touchdown passes did \w+ \w+ \w+ in the \w+ \w+ \?",
    "how many passes did \w+ \w+ in the \w+ \w+ \?",
    "how many passes did \w+ \w+ \w+ in the \w+ \w+ \?",
    "how many field goals did \w+ \w+ in the \w+ \w+ \?",
    "how many field goals did \w+ \w+ \w+ in the \w+ \w+ \?",
]


COUNT_FIND_REGEX = re.compile("|".join(COUNT_FIND_QUESTION_REGEX_PATTERNS))
COUNT_FILTER_FIND_REGEX = re.compile("|".join(COUNT_FILTER_FIND_QUESTION_REGEX_PATTERNS))


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def is_count_find_question(tokenized_question_lower: str):
    match_result = COUNT_FIND_REGEX.fullmatch(tokenized_question_lower)
    if match_result is not None:
        return True
    else:
        return False


def is_count_filter_find_question(tokenized_question_lower: str):
    match_result = COUNT_FILTER_FIND_REGEX.fullmatch(tokenized_question_lower)
    if match_result is not None:
        return True
    else:
        return False


def convert_start_end_to_attention_vector(length, start, end):
    """Convert start/end of a span into a binary vector. start/end is inclusive/exclusive."""
    attention_vector = [0.0] * length
    attention_vector[start:end] = [1.0] * (end - start)
    return attention_vector


def get_count_find_question_attention(question_tokens: List[str]):
    """Compute the find question attention for count-find questions.

    These are of the kind "how many field goals did \w+ \w+ \w+ \?" where every token after "how many" is find-arg
    """
    len_question = len(question_tokens)
    find_start = 2
    find_end = len_question - 1  # exclusive, don't attend to ?
    find_attention_vector = convert_start_end_to_attention_vector(len_question, find_start,
                                                                  find_end)
    return find_attention_vector


def get_count_filter_find_question_attention(question_tokens: List[str]):
    """Compute the find question attention for count-find questions.

    These are of the kind "how many passes did \w+ \w+ in/during the \w+ \w+ \?"
        filter_attention: last 3 tokens "the \w+ \w+"
        find_attention: token after "how many" and before "in/during"
    """
    len_question = len(question_tokens)
    filter_start = len_question - 4  # inclusive, the last 3 tokens excluding "?"
    filter_end = len_question - 1  # exclusive, last token before "?"

    find_end = filter_start - 1  # exclusive, need to skip "in" before filter-start but only -1 since end is exclusive
                                 #  and start is inclusive
    find_start = 2  # inclusive, token after "how many"

    find_attention_vector = convert_start_end_to_attention_vector(len_question, find_start, find_end)
    filter_attention_vector = convert_start_end_to_attention_vector(len_question, filter_start, filter_end)
    return filter_attention_vector, find_attention_vector


def preprocess_HowManyYardsCount_ques(dataset):
    """ This function prunes for questions that are count based questions.

        Along with pruning, we also supervise the with the qtype and program_supervised flag
    """

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    num_passages = len(dataset)
    prog_dist = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()
            tokenized_ques = question_answer[constants.tokenized_question]
            question_tokens = tokenized_ques.split(" ")
            if any(span in question_lower for span in COUNT_TRIGRAMS):
                is_count_find = is_count_find_question(tokenized_ques.lower())
                if is_count_find:
                    qtype = constants.COUNT_find_qtype
                    question_answer[constants.qtype] = qtype
                    question_answer[constants.program_supervised] = True
                    find_qattn = get_count_find_question_attention(question_tokens)
                    question_answer[constants.ques_attention_supervision] = [find_qattn]
                    question_answer[constants.qattn_supervised] = True
                    prog_dist[qtype] += 1

                is_count_filter_find = is_count_filter_find_question(tokenized_ques.lower())
                if is_count_filter_find:
                    qtype = constants.COUNT_filter_find_qtype
                    question_answer[constants.qtype] = qtype
                    question_answer[constants.program_supervised] = True
                    filter_qattn, find_qattn = get_count_filter_find_question_attention(question_tokens)
                    question_answer[constants.ques_attention_supervision] = [filter_qattn, find_qattn]
                    question_answer[constants.qattn_supervised] = True
                    prog_dist[qtype] += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"Passages after-pruning:{num_passages_after_prune}  Question after-pruning:{after_pruning_ques}")
    print(f"Program distribution: {prog_dist}")

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput dir: {output_dir}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = preprocess_HowManyYardsCount_ques(train_dataset)

    new_dev_dataset = preprocess_HowManyYardsCount_ques(dev_dataset)

    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written count dataset")
