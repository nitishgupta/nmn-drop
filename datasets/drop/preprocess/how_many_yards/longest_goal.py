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
from semqa.domain_languages.drop.drop_language import Date
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(["'s", ","])

FIRST = "first"
SECOND = "second"

DATE_COMPARISON_TRIGRAMS = ["which happened", "which event", "what happened first", "what happened second",
                            "what happened later", "what happened last", "what event happened", "what event came"]


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def preprocess_HowManyYardsLongestShortestQues(dataset):
    """ This function foccuses on questions that start with "How many yards was the". These can come directly from the
        preprocessed dataset, or from the corresponding ngram-slice; aka we test for this n-gram in this script as well.

        For a few question types given below, we add a corresponding qtype key and strongly_supervised flag to them
        This could act as gold-program auxiliary supervision in the model

        How many yards was the longest touchdown?
        How many yards was the longest touchdown pass?" -- 349 examples
        How many yards was the longest field goal?"

        How many yards was the shortest touchdown?
        How many yards was the shortest touchdown pass?
        How many yards was the shortest field goal?

        How many yards was the second longest touchdown pass? -- 27
        How many yards was the second longest touchdown? -- 38
        How many yards was the second longest field goal? -- 37

        How many yards was the second shortest touchdown? -- 7
        How many yards was the second shortest touchdown pass?
        How many yards was the second shortest field goal? -- 10

        qtype for `longest = how_many_longest`; `shortest = how_many_shortest`;
        `second longest = how_many_second_longest` and `second shortest = how_many_second_shortest`
    """

    longest_questions = ["How many yards was the longest touchdown?",
                         "How many yards was the longest touchdown pass?",
                         "How many yards was the longest field goal?"]

    shortest_questions = ["How many yards was the shortest touchdown?",
                          "How many yards was the shortest touchdown pass?",
                          "How many yards was the shortest field goal?"]

    second_longest_questions = ["How many yards was the second longest touchdown pass?",
                                "How many yards was the second longest touchdown?",
                                "How many yards was the second longest field goal?"]

    second_shortest_questions = ["How many yards was the second shortest touchdown?",
                                 "How many yards was the second shortest touchdown pass?",
                                 "How many yards was the second shortest field goal?"]

    question_start_ngram = "How many yards was the"

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

            if question_start_ngram.lower() in original_question.lower():
                if original_question in longest_questions:
                    question_answer[constants.qtype] = longest_qtype
                    question_answer[constants.strongly_supervised] = True
                    qtype_dist[longest_qtype] += 1
                    questions_w_qtypes += 1

                elif original_question in shortest_questions:
                    question_answer[constants.qtype] = shortest_qtype
                    question_answer[constants.strongly_supervised] = True
                    qtype_dist[shortest_qtype] += 1
                    questions_w_qtypes += 1

                elif original_question in second_longest_questions:
                    question_answer[constants.qtype] = second_longest_qtype
                    question_answer[constants.strongly_supervised] = True
                    qtype_dist[second_longest_qtype] += 1
                    questions_w_qtypes += 1

                elif original_question in second_shortest_questions:
                    question_answer[constants.qtype] = second_shortest_qtype
                    question_answer[constants.strongly_supervised] = True
                    qtype_dist[second_shortest_qtype] += 1
                    questions_w_qtypes += 1
                else:
                    question_answer[constants.strongly_supervised] = False


                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with qtypes and strongly supervised: {questions_w_qtypes}")
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

    input_dir = "./resources/data/drop/analysis/ngram/num/how_many_yards_was_the/"
    output_dir = "./resources/data/drop/num/how_many_yards_was_the"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output dir: {output_dir}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = preprocess_HowManyYardsLongestShortestQues(train_dataset)

    new_dev_dataset = preprocess_HowManyYardsLongestShortestQues(dev_dataset)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written augmented datasets")

''' DATASET CREATED THIS WAY

input_dir = "./resources/data/drop/analysis/ngram/num/how_many_yards_was_the/"
output_dir = "./resources/data/drop/num/how_many_yards_was_the"

'''
