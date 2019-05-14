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
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(["'s", ","])

FIRST = "first"
SECOND = "second"

RELEVANT_NGRAMS = ["how many yards was", "how many yards longer was", "how many yards difference",
                   "how many field goals did", "how many field goals were", "how many touchdowns were scored",
                   "how many touchdowns did the"]


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def preprocess_HowManyYardsWasThe_ques(dataset):
    """ This function prunes questions that start with "How many yards was".
        Mostly, longest, shortest style questions. We can also prune for these; look at longestshortest_ques.py

        Along with pruning, we also supervise the longest/shortest/second longest/second shortest questions
        by adding the question_type for those questions.
    """

    longest_question_ngram = "How many yards was the longest"
    shortest_question_ngram = "How many yards was the shortest"
    second_longest_question_ngram = "How many yards was the second longest"
    second_shortest_question_ngram = "How many yards was the second shortest"

    longest_qtype = 'how_many_yards_longest'
    shortest_qtype = 'how_many_yards_shortest'
    second_longest_qtype = 'how_many_yards_second_longest'
    second_shortest_qtype = 'how_many_yards_second_shortest'

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_qtypes = 0
    qtype_dist = defaultdict(int)
    num_passages = len(dataset)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()

            if any(span in question_lower for span in RELEVANT_NGRAMS):
                if longest_question_ngram in original_question:
                    question_answer[constants.qtype] = constants.YARDS_longest_qtype
                    question_answer[constants.program_supervised] = True
                    qtype_dist[longest_qtype] += 1
                    questions_w_qtypes += 1

                elif shortest_question_ngram in original_question:
                    question_answer[constants.qtype] = constants.YARDS_shortest_qtype
                    question_answer[constants.program_supervised] = True
                    qtype_dist[shortest_qtype] += 1
                    questions_w_qtypes += 1

                # elif second_longest_question_ngram in original_question:
                #     question_answer[constants.qtype] = constants.YARDS_second_longest_qtype
                #     question_answer[constants.program_supervised] = True
                #     qtype_dist[second_longest_qtype] += 1
                #     questions_w_qtypes += 1
                #
                # elif second_shortest_question_ngram in original_question:
                #     question_answer[constants.qtype] = constants.YARDS_second_shortest_qtype
                #     question_answer[constants.program_supervised] = True
                #     qtype_dist[second_shortest_qtype] += 1
                #     questions_w_qtypes += 1
                else:
                    question_answer[constants.program_supervised] = False

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with qtypes and program supervised: {questions_w_qtypes}")
    print(f"Qtype dist: {qtype_dist}")

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

    input_dir = "./resources/data/drop_s/preprocess"
    output_dir = "./resources/data/drop_s/num/how_many_nums"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output dir: {output_dir}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = preprocess_HowManyYardsWasThe_ques(train_dataset)

    new_dev_dataset = preprocess_HowManyYardsWasThe_ques(dev_dataset)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written pruned datasets")

''' DATASET CREATED THIS WAY

input_dir = "./resources/data/drop_old/analysis/ngram/num/how_many_yards_was_the/"
output_dir = "./resources/data/drop_old/num/how_many_yards_was_the"

'''
