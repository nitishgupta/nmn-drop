import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union
import random
from datasets.drop import constants
from semqa.utils.qdmr_utils import node_from_dict, Node, read_drop_dataset

""" 
    Remove Execution Supervised key from each question in the Train data (only)
"""

execution_supervision_keys = ["date1_entidxs", "date2_entidxs", "num1_entidxs", "num2_entidxs",
                              "num_entidxs"]

keys_not_to_remove = ["question_attention_supervision"]

def remove_execution_supervision(node: Node):
    supervision_dict: Dict = node.supervision
    supervision_dict_keys = list(supervision_dict.keys())
    for key in supervision_dict_keys:
        if key in execution_supervision_keys:
            supervision_dict.pop(key, None)
        elif key not in keys_not_to_remove:
            # Some supervision key we don't recognize
            print(f"Unrecognized supervision_dict key: {key}")

    node.supervision = supervision_dict

    new_children = []
    for c in node.children:
        c_new = remove_execution_supervision(c)
        new_children.append(c_new)

    node.children = []
    for c_new in new_children:
        node.add_child(c_new)

    return node


def removeExecutionSupervision(dataset):
    """ Given a dataset, remove execution supervision from all questions. """

    total_num_passages = len(dataset)
    total_num_qa = 0
    num_exec_sup = 0

    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                program_node: Node = node_from_dict(qa[constants.program_supervision])
                new_program_node = remove_execution_supervision(program_node)   # what-if exec-supervised is not set
                qa[constants.program_supervision] = new_program_node.to_dict()

                execution_supervised = qa.get(constants.execution_supervised, None)
                if execution_supervised is not None:
                    num_exec_sup += 1 if execution_supervised else 0
                    qa.pop(constants.execution_supervised)

    print(f"TotalNumPassages: {total_num_passages}")
    print(f"Num of original question: {total_num_qa} num-exec-sup: {num_exec_sup}")

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

    print(f"\nRemoving intermediate execution supervision from:\nD1: {input_dir}\nOutDir:{output_dir}\n")

    FILES_TO_CLEAN = ["drop_dataset_train.json", "drop_dataset_dev.json", "drop_dataset_test.json"]

    for filename in FILES_TO_CLEAN:
        print(filename)
        input_json = os.path.join(input_dir, filename)
        output_json = os.path.join(output_dir, filename)

        if not os.path.exists(input_json):
            print(f"{input_json} does not exist. Skipping.")
            continue

        input_dataset = read_drop_dataset(input_json)
        output_dataset = removeExecutionSupervision(input_dataset)

        print(f"\nWriting output: {output_json}")
        with open(output_json, "w") as f:
            json.dump(output_dataset, f, indent=4)