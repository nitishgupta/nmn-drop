from typing import List, Dict, Tuple
import json
from nltk.corpus import stopwords
import os
import copy
import torch
from allennlp.models.reading_comprehension.util import get_best_span
import allennlp.nn.util as allenutil
from collections import defaultdict
import datasets.drop.constants as constants
from semqa.domain_languages.drop_old.drop_language import Date
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(["'s", ","])

FIRST = "first"
SECOND = "second"

COUNT_TRIGRAMS = ["how many field goals did", "how many field goals were",
                  "how many interceptions did", "how many passes", "how many rushing",
                  "how many touchdown passes did", "how many touchdowns did the",
                  "how many touchdowns were scored"]


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def preprocess_HowManyYardsCount_ques(dataset):
    """ This function prunes for questions that are count based questions.

        Along with pruning, we also supervise the with the qtype and program_supervised flag
    """

    count_qtype = constants.COUNT_qtype

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_qtypes = 0
    num_passages = len(dataset)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()
            tokenized_ques = question_answer[constants.tokenized_question]
            tokens = tokenized_ques.split(' ')
            if any(span in question_lower for span in COUNT_TRIGRAMS):
                question_answer[constants.qtype] = constants.COUNT_qtype
                question_answer[constants.program_supervised] = True
                questions_w_qtypes += 1

                # Also adding qattn -- everything apart from the first two tokens
                attention = [1.0 if x > 1 else 0.0 for x in range(len(tokens))]
                question_answer[constants.ques_attention_supervision] = [attention]
                question_answer[constants.qattn_supervised] = True

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"Passages after-pruning:{num_passages_after_prune}  Question after-pruning:{after_pruning_ques}")
    print(f"Num of QA with qtypes and program supervised: {questions_w_qtypes}")

    return new_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

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

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written count dataset")

