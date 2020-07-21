import os
import re
import json
import copy
import random
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable

from analysis.qdmr.program_diagnostics import is_potential_filter_num
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    lisp_to_nested_expression, nested_expression_to_tree, get_postorder_function_list, \
    get_postorder_function_and_arg_list, read_jsonl
from semqa.domain_languages.drop_language import DropLanguage, get_empty_language_object
from allennlp.data.tokenizers import SpacyTokenizer

from datasets.drop import constants

random.seed(42)

spacy_tokenizer = SpacyTokenizer()

nmndrop_language: DropLanguage = get_empty_language_object()


class QuesParsePrediction():
    def __init__(self, prediction_dict: Dict):
        self.question = prediction_dict["question"]
        self.query_id = prediction_dict["query_id"]
        self.gold_logical_form = prediction_dict["gold_logical_form"]
        self.gold_nested_expr = prediction_dict["gold_nested_expr"]
        self.gold_nested_expr_wstr = prediction_dict["gold_nested_expr_wstr"]
        self.top_logical_form = prediction_dict["top_logical_form"]
        self.top_nested_expr_wstr = prediction_dict["top_nested_expr_wstr"]
        self.top_logical_form_prob = prediction_dict["top_logical_form_prob"]
        self.top_program_dict = prediction_dict["top_program_dict"]
        self.top_program_node: Node = node_from_dict(self.top_program_dict)


def read_qparse(prediction_dict: Dict):
    if prediction_dict["top_logical_form"] == "":
        return None
    return QuesParsePrediction(prediction_dict)


def get_predicted_string_args(node: Node):
    string_args = []
    predicted_string_arg = node.extras.get("predicted_ques_string_arg", None)
    if predicted_string_arg is not None and node.predicate == "select_passage":
        string_args.append(predicted_string_arg)
    for c in node.children:
        string_args.extend(get_predicted_string_args(c))

    return string_args


def get_stringarg2idxs(ques_parses: List[QuesParsePrediction]):
    stringarg2idxs = defaultdict(list)
    for idx, ques_parse in enumerate(ques_parses):
        string_args = get_predicted_string_args(ques_parse.top_program_node)
        for stringarg in string_args:
            stringarg2idxs[stringarg].append(idx)
    return stringarg2idxs


def print_common(ques_parses: List[QuesParsePrediction]):
    count = 0
    stringarg2idxs = get_stringarg2idxs(ques_parses)

    for stringarg, idxs in stringarg2idxs.items():
        if len(set(idxs)) <= 1:
            continue
        count += 1
        print(stringarg)
        print([ques_parses[idx].question for idx in idxs])
        print()

    print("Total string args: {}".format(count))


def main(args):
    # input_dir = args.input_dir

    ques_parse_dicts: List[Dict] = read_jsonl(args.input_jsonl)

    ques_parses: List[QuesParsePrediction] = []
    for d in ques_parse_dicts:
        qparse: QuesParsePrediction = read_qparse(d)
        if qparse is not None:
            ques_parses.append(qparse)

    print("Len: {}".format(len(ques_parses)))
    print_common(ques_parses)

    # output_dir = args.output_dir
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #
    # prune_dataset: bool = args.prune_dataset
    #
    # FILES_TO_FILTER = ["drop_dataset_train.json", "drop_dataset_dev.json"]
    # stats = ""
    #
    # for filename in FILES_TO_FILTER:
    #     stats += "Stats for : {}\n".format(filename)
    #
    #     input_json = os.path.join(input_dir, filename)
    #     print("Reading data from: {}".format(input_json))
    #     dataset = read_drop_dataset(input_json)
    #     dataset_w_sharedsub, file_stats = get_postprocessed_dataset(dataset=dataset, prune_dataset=prune_dataset)
    #
    #     stats += file_stats + "\n"
    #
    #     output_json = os.path.join(args.output_dir, filename)
    #     print(f"OutFile: {output_json}")
    #
    #     print(f"Writing data w/ shared-substructures to: {output_json}")
    #     with open(output_json, 'w') as outf:
    #         json.dump(dataset_w_sharedsub, outf, indent=4)
    #     print()
    #
    # stats_file = os.path.join(output_dir, "stats.txt")
    # print(f"\nWriting stats to: {stats_file}")
    # with open(stats_file, "w") as outf:
    #     outf.write(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl")
    # parser.add_argument("--output_dir")
    # parser.add_argument('--prune-dataset', dest='prune_dataset', action='store_true')

    args = parser.parse_args()

    main(args)

