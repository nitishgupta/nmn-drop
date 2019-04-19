import os
import sys
import copy
import time
import json
import string
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from utils import util, spacyutils
from datasets.drop import constants
from datasets.drop.preprocess import ner_process

import multiprocessing

DATE_COMPARISON = ["which happened", "which event", "what happened first", "what happened second",
                   "what happened later", "what happened last", "what event happened", "what event came"]

NUMBER_COMPARISON = ["were there more", "were there fewer", "which age group", "which group", "how long was",
                     "who caught", "who threw", "who kicked", "who scored the"]

def date_comparison_filter(question: str):
    if any(span in question for span in DATE_COMPARISON):
        return True
    else:
        return False

def number_comparison_filter(question: str):
    if any(span in question for span in NUMBER_COMPARISON):
        return True
    else:
        return False


def processPassage(passage_info, keep_date: bool, keep_num: bool):
    """ Keep relevant qa_pairs from the passage info.

        NOTE: Only keeping qa that have span-based answers
    """

    assert keep_date or keep_num, "Atleast one should be true"

    qa_pairs = passage_info[constants.qa_pairs]
    relevant_qa_pairs = []

    for qa_pair in qa_pairs:
        keep = False
        question = qa_pair[constants.tokenized_question].lower()

        if date_comparison_filter(question) and keep_date:
            keep = True

        if number_comparison_filter(question) and keep_num:
            keep = True

        if constants.answer_type in qa_pair:
            if qa_pair[constants.answer_type] != constants.SPAN_TYPE:
                keep = False
        else:
            keep = False

        # To avoid duplication
        if keep:
            relevant_qa_pairs.append(qa_pair)


    if len(relevant_qa_pairs) == 0:
        return None

    passage_info[constants.qa_pairs] = relevant_qa_pairs

    return passage_info


def pruneDataset(input_json: str, output_json: str, keep_date: bool, keep_num: bool) -> None:
    """ Prune dataset to only contain questions that qualify after certain tests.
        Currently only keeping questions with a SpanType answer.
    """

    print("Reading input json: {}".format(input_json))
    print("Output filepath: {}".format(output_json))

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json, 'r') as f:
        dataset = json.load(f)

    print("Number of docs: {}".format(len(dataset)))

    numdocswritten = 0

    num_input_qas = 0

    # List of tuples with (passage_id, passage_info)
    passage_id_infos = list(dataset.items())
    for (_, pinfo) in passage_id_infos:
        num_input_qas += len(pinfo[constants.qa_pairs])

    output_passage_dict = {}
    num_output_qas = 0
    for passage_id, passage_info in dataset.items():
        new_p_info = processPassage(passage_info, keep_date, keep_num)
        if new_p_info is None:
            continue
        output_passage_dict[passage_id] = new_p_info
        num_output_qas += len(new_p_info[constants.qa_pairs])

    with open(output_json, 'w') as outf:
        json.dump(output_passage_dict, outf, indent=4)

    print(f"Number of QA pairs input: {num_input_qas}")
    print(f"Number of QA pairs output: {num_output_qas}")
    print(f"Total docs output: {len(output_passage_dict)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_json', default=True)
    parser.add_argument('--keep_date', action='store_true', default=False)
    parser.add_argument('--keep_num', action='store_true', default=False)
    args = parser.parse_args()

    keep_date = args.keep_date
    keep_num= args.keep_num

    # args.input_json --- is the raw json from the DROP dataset
    pruneDataset(input_json=args.input_json, output_json=args.output_json, keep_date=keep_date, keep_num=keep_num)
