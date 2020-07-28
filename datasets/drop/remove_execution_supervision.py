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


def remove_execution_supervision(node: Node):
    supervision_dict: Dict = node.supervision
    for key in execution_supervision_keys:
        supervision_dict.pop(key, None)

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
                execution_supervised = qa.get(constants.execution_supervised, None)
                if execution_supervised is not None:
                    num_exec_sup += 1 if execution_supervised else 0
                    qa[constants.execution_supervised] = False

                    new_program_node = remove_execution_supervision(program_node)
                    qa[constants.program_supervision] = new_program_node.to_dict()

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

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = read_drop_dataset(input_trnfp)
    dev_dataset = read_drop_dataset(input_devfp)

    print("\nTraining questions .... ")
    new_train_dataset = removeExecutionSupervision(train_dataset)

    print("\nDev questions .... ")
    new_dev_dataset = removeExecutionSupervision(dev_dataset)

    print(f"\nOutput path: {output_trnfp}")
    with open(output_trnfp, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    print(f"\nOutput path: {output_devfp}")
    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)
