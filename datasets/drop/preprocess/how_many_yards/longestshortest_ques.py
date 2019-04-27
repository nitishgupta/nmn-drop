from typing import List, Dict, Tuple
import json
import os
import copy
import torch
from allennlp.models.reading_comprehension.util import get_best_span
import allennlp.nn.util as allenutil
from collections import defaultdict
import datasets.drop.constants as constants
from semqa.domain_languages.drop.drop_language import Date
import argparse


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def preprocess_HowManyYardsLongestShortestQues(dataset):
    """ This function takes the output of the how_many_yards.py (containing qtype-supervision)
        and prunes only the longest / shortest question to test max_distribution / min_distribution

        This also adds question_attention supervision. Attention=1.0 for "touchdown of the second half" in
        "How many yards was the second longest touchdown of the second half?"
    """

    starting_ngrams = ["How many yards was the longest", "How many yards was the shortest"]

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    num_passages = len(dataset)

    supervision_dict = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question = question_answer[constants.tokenized_question]

            if any(span in original_question for span in starting_ngrams):
                question_tokens = question.split()
                qlen = len(question_tokens)

                # How many yards was the longest/shortest has 6 tokens and excluding question mark at the end
                attention_vec = [0.0 if i < 6 or i == qlen-1 else 1.0 for i in range(qlen)]
                if sum(attention_vec) == 0:
                    continue

                # This is a list of question attentions for the possibly many predicates that take
                # question_attention as a side_arg in the correct program
                question_answer[constants.ques_attention_supervision] = [attention_vec]
                question_answer[constants.qattn_supervised] = True

                new_qa_pairs.append(question_answer)

        for qa in new_qa_pairs:
            if constants.program_supervised in qa:
                supervision_dict[constants.program_supervised] += 1 if qa[constants.program_supervised] else 0
            if constants.qattn_supervised in qa:
                supervision_dict[constants.qattn_supervised] += 1 if qa[constants.qattn_supervised] else 0
            if constants.exection_supervised in qa:
                supervision_dict[constants.program_supervised] += 1 if qa[constants.exection_supervised] else 0

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(supervision_dict)

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
