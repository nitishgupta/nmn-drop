import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable

from utils.util import _KnuthMorrisPratt
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp

from datasets.drop import constants

import itertools

FILTER_MODIFIERS = [
    "the first quarter",
    "the second quarter",
    "the third quarter",
    "the fourth quarter",
    "the first half",
    "the second half",
    "the first two quarters",
    "the last two quarters",
]

PREPOSITIONS = ["of", "in", "after", "before"]

FILTER_MODIFIERS_FINAL = []
for (x, y) in itertools.product(PREPOSITIONS, FILTER_MODIFIERS):
    FILTER_MODIFIERS_FINAL.append(f"{x} {y}")


def find_sublist(sublist, parent_list):
    # In ICLR21 dataset, we don't match only about ~18-20 questions
    idxs = list(_KnuthMorrisPratt(parent_list, sublist))
    if not idxs:
        # print("{} not found in {}".format(sublist, parent_list))
        return None
    elif len(idxs) > 1:
        # print("Multiple {} found in {}".format(sublist, parent_list))
        return idxs[0]
        # return None
    else:
        return idxs[0]

QTYPE2COUNT = {"minmax": 0, "project": 0, "count": 0}

def filter_to_known_programs(qdmr_node: Node, question: str, question_tokens: str):
    global QTYPE2COUNT
    program_lisp = nested_expression_to_lisp(qdmr_node.get_nested_expression())
    change = 0
    if program_lisp not in ["(select_num (select_max_num select_passage))",
                            "(select_num (select_min_num select_passage))",
                            "(select_num select_passage)",
                            "(select_passagespan_answer (project_passage (select_max_num select_passage)))",
                            "(select_passagespan_answer (project_passage (select_min_num select_passage)))",
                            "(aggregate_count select_passage)"]:
        return qdmr_node, change

    if any([span in question for span in FILTER_MODIFIERS_FINAL]):
        # Assume that the last 4 tokens are filter tokens; add filter accordingly
        matched_span = None
        for span in FILTER_MODIFIERS_FINAL:
            if span in question:
                matched_span = span
                break

        qlen = len(question_tokens)
        filter_arg = matched_span
        filter_qattn = [0 for _ in range(qlen)]
        for i in range(0, 4):
            filter_qattn[qlen - 1 - 1 - i] = 1  # extra -1 to avoid '?' at the end

        filter_node = Node(predicate="filter_passage")
        filter_node.string_arg = filter_arg
        filter_node.supervision["question_attention_supervision"] = filter_qattn

        if program_lisp in ["(select_num (select_max_num select_passage))",
                            "(select_num (select_min_num select_passage))"]:
            minmax_node = qdmr_node.children[0]
            old_select_node = minmax_node.children[0]
            if "question_attention_supervision" in old_select_node.supervision:
                select_qattn = old_select_node.supervision["question_attention_supervision"]
                for i in range(0, 4):
                    select_qattn[qlen - 1 - 1 - i] = 0  # extra -1 to avoid '?' at the end
                old_select_node.supervision["question_attention_supervision"] = select_qattn
            filter_node.add_child(old_select_node)
            minmax_node.children = []
            minmax_node.add_child(filter_node)
            change = 1
            QTYPE2COUNT["minmax"] += 1
        elif program_lisp in ["(select_passagespan_answer (project_passage (select_max_num select_passage)))",
                              "(select_passagespan_answer (project_passage (select_min_num select_passage)))"]:
            minmax_node = qdmr_node.children[0].children[0]
            old_select_node = minmax_node.children[0]
            if "question_attention_supervision" in old_select_node.supervision:
                select_qattn = old_select_node.supervision["question_attention_supervision"]
                for i in range(0, 4):
                    select_qattn[qlen - 1 - 1 - i] = 0  # extra -1 to avoid '?' at the end
                old_select_node.supervision["question_attention_supervision"] = select_qattn
            filter_node.add_child(old_select_node)
            minmax_node.children = []
            minmax_node.add_child(filter_node)
            change = 1
            QTYPE2COUNT["project"] += 1
        elif program_lisp in ["(aggregate_count select_passage)",
                              "(select_num select_passage)"]:
            old_select_node = qdmr_node.children[0]
            if "question_attention_supervision" in old_select_node.supervision:
                select_qattn = old_select_node.supervision["question_attention_supervision"]
                for i in range(0, 4):
                    select_qattn[qlen - 1 - 1 - i] = 0  # extra -1 to avoid '?' at the end
                old_select_node.supervision["question_attention_supervision"] = select_qattn
            filter_node.add_child(old_select_node)
            qdmr_node.children = []
            qdmr_node.add_child(filter_node)
            change = 1
            QTYPE2COUNT["count"] += 1

    return qdmr_node, change


def add_filter_node(qdmr_node: Node, question: str, question_tokens: List[str]):
    change = 0
    if qdmr_node.predicate == "select_passage":
        string_arg = qdmr_node.string_arg
        if string_arg is not None and any([span in string_arg for span in FILTER_MODIFIERS_FINAL]):
            # Add filter module. Attention on second to last 3 tokens
            qlen = len(question_tokens)
            matched_span = None
            for span in FILTER_MODIFIERS_FINAL:
                if span in string_arg:
                    matched_span = span
                    break
            select_arg = string_arg.replace(matched_span, "").strip()
            filter_arg = matched_span
            filter_tokens = filter_arg.split(" ")
            # Find this sub-sequence in the question tokens
            starting_idx = find_sublist(filter_tokens, question_tokens)
            if starting_idx is None:
                filter_tokens[0] = "in"
                starting_idx = find_sublist(filter_tokens, question_tokens)
                # if starting_idx is None:
                #     import pdb
                #     pdb.set_trace()
            if starting_idx is not None:
                select_qattn = None
                if "question_attention_supervision" in qdmr_node.supervision:
                    select_qattn = qdmr_node.supervision["question_attention_supervision"]
                filter_qattn = [0 for _ in range(qlen)]
                for i in range(starting_idx, starting_idx + len(filter_tokens)):
                    filter_qattn[i] = 1
                    if select_qattn is not None:
                        select_qattn[i] = 0

                filter_node = Node(predicate="filter_passage")
                filter_node.string_arg = filter_arg
                filter_node.supervision["question_attention_supervision"] = filter_qattn

                select_node = Node(predicate="select_passage")
                select_node.string_arg = select_arg
                select_node.supervision = qdmr_node.supervision
                if select_qattn is not None:
                    select_node.supervision["question_attention_supervision"] = select_qattn

                filter_node.add_child(select_node)
                filter_node.parent = qdmr_node.parent
                qdmr_node = filter_node

                change = 1

    new_children = []
    for child in qdmr_node.children:
        new_child, x = add_filter_node(child, question, question_tokens)
        new_children.append(new_child)
        change = min(1, change + x)

    qdmr_node.children = []
    for c in new_children:
        qdmr_node.add_child(c)
    return qdmr_node, change


def get_postprocessed_dataset(dataset: Dict) -> Dict:
    """ Add filter module for "Nth quarter / ha;f questions """
    total_qa = 0
    num_converted_qa = 0

    qtype2conversion = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        for qa in passage_info[constants.qa_pairs]:
            question = qa[constants.question]
            question_tokens = qa[constants.question_tokens]
            total_qa += 1
            if constants.program_supervision not in qa:
                continue

            else:
                program_node = node_from_dict(qa[constants.program_supervision])

                post_processed_node = copy.deepcopy(program_node)
                post_processed_node, change = add_filter_node(post_processed_node, question, question_tokens)
                if not change:
                    post_processed_node, change = filter_to_known_programs(post_processed_node, question,
                                                                           question_tokens)
                if change:
                    qtype2conversion["add_filter"] += 1
                    # print()
                    # print(question)
                    # print(program_node.to_dict())
                    # print(post_processed_node.to_dict())

                qa["preprocess_program_supervision"] = program_node.to_dict()
                qa[constants.program_supervision] = post_processed_node.to_dict()

    print()
    print(f"Number of input passages: {len(dataset)}\nNumber of input questions: {total_qa}")
    print(f"QType 2 conversion count: {qtype2conversion}")
    print(QTYPE2COUNT)
    return dataset


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    FILES_TO_FILTER = ["drop_dataset_train.json", "drop_dataset_dev.json", "drop_dataset_test.json"]

    for filename in FILES_TO_FILTER:
        print(filename)
        input_json = os.path.join(args.input_dir, filename)
        print(f"Input json: {input_json}")

        if not os.path.exists(input_json):
            print("File does not exist: {}".format(input_json))
            continue

        postprocessed_dataset = get_postprocessed_dataset(dataset=read_drop_dataset(input_json))

        output_json = os.path.join(args.output_dir, filename)
        print(f"OutFile: {output_json}")

        print(f"Writing post-processed data to : {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(postprocessed_dataset, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    main(args)

