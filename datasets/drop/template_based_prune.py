import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union
import random
from datasets.drop import constants
from semqa.utils.qdmr_utils import node_from_dict, Node, read_drop_dataset, nested_expression_to_lisp

""" 
    Remove Execution Supervised key from each question in the Train data (only)
"""


def pruneDataset(dataset, keep_templates: List[str]):
    """ Given a dataset, remove execution supervision from all questions. """

    total_num_passages = len(dataset)
    total_num_qa = 0
    remain_q, remain_p = 0, 0

    paras_to_remove = []
    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])
        pruned_qas = []
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                program_node: Node = node_from_dict(qa[constants.program_supervision])
                program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())

                if program_lisp in keep_templates:
                    pruned_qas.append(qa)
                    remain_q += 1

        if not pruned_qas:
            paras_to_remove.append(passage_idx)
        passage_info[constants.qa_pairs] = pruned_qas

    for pid in paras_to_remove:
        dataset.pop(pid)

    print(f"Input NumPassages: {total_num_passages}  NumQ:{total_num_qa}")
    print(f"After pruning; Passages: {len(dataset)} NumQ: {remain_q}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    keep_templates = ["(select_num (select_max_num select_passage))", "(select_num (select_min_num select_passage))"]
    print(f"")
    print(f"\nPruning examples from:\nInDir1: {input_dir}\nOutDir:{output_dir}\n")
    print(f"Keeping only templates: {keep_templates}")

    FILES_TO_CLEAN = ["drop_dataset_train.json", "drop_dataset_dev.json", "drop_dataset_test.json"]

    for filename in FILES_TO_CLEAN:
        print(filename)
        input_json = os.path.join(input_dir, filename)
        output_json = os.path.join(output_dir, filename)

        if not os.path.exists(input_json):
            print(f"{input_json} does not exist. Skipping.")
            continue

        input_dataset = read_drop_dataset(input_json)
        output_dataset = pruneDataset(input_dataset, keep_templates)

        print(f"Writing output: {output_json}\n")
        with open(output_json, "w") as f:
            json.dump(output_dataset, f, indent=4)