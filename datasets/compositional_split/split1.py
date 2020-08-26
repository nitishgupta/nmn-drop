from typing import List, Tuple, Dict, Union
import os
import json
import copy
import argparse
from collections import defaultdict

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp
from datasets.drop import constants


def prune_dataset(dataset: Dict, keep: bool, modules: List[str]) -> Dict:
    """Prune questions based on existence (or not) of the given modules.

    Parameters:
    ----------
    keep: `bool`
        If True, keep question iff one of the modules from the list appears in the question-program
        If False, question shouldn't contain any of the modules mentioned
    modules: `List[str]`
        List of module names that govern the pruning.
    """

    print("\nPruning dataset based on modules: {} and keep: {}".format(modules, keep))

    pruned_dataset = {}
    num_input_passage, num_output_passage = 0, 0
    num_input_questions, num_output_questions = 0, 0

    for pid, passage_info in dataset.items():
        num_input_passage += 1
        pruned_qas = []
        for qa in passage_info[constants.qa_pairs]:
            num_input_questions += 1
            if constants.program_supervision not in qa or not qa[constants.program_supervision]:
                continue
            program_node = node_from_dict(qa[constants.program_supervision])
            program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())

            # If keep=True, add question if any of the modules is found
            if keep and any([module in program_lisp for module in modules]):
                pruned_qas.append(qa)
                num_output_questions += 1

            # If keep=False, only keep iff any of the modules is not found
            if not keep and all([module not in program_lisp for module in modules]):
                pruned_qas.append(qa)
                num_output_questions += 1

        if pruned_qas:
            passage_info[constants.qa_pairs] = pruned_qas
            pruned_dataset[pid] = passage_info
            num_output_passage += 1


    print("Num of input passage: {}  question: {}".format(num_input_passage, num_input_questions))
    print("Num of output passage: {}  question: {}".format(num_output_passage, num_output_questions))

    return pruned_dataset


def merge_datasets(train_dataset, dev_dataset):
    merged_dataset = {}
    merged_dataset.update(train_dataset)
    merged_dataset.update(dev_dataset)
    return merged_dataset


def main(args):
    input_dir = args.input_dir

    train_drop_json = os.path.join(input_dir, "drop_dataset_train-wcontrastive.json")
    dev_drop_json = os.path.join(input_dir, "drop_dataset_dev.json")

    train_drop = read_drop_dataset(train_drop_json)
    dev_drop = read_drop_dataset(dev_drop_json)

    modules_for_dev = ["compare_date_lt", "compare_num_lt", "select_min_num"]

    new_train_dataset = prune_dataset(train_drop, keep=False, modules=modules_for_dev)
    new_dev_dataset = prune_dataset(dev_drop, keep=True, modules=modules_for_dev)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_train_drop_json = os.path.join(output_dir, "drop_dataset_train-wcontrastive.json")
    output_dev_drop_json = os.path.join(output_dir, "drop_dataset_dev.json")

    print(f"\nWriting train data to : {output_train_drop_json}")
    with open(output_train_drop_json, 'w') as outf:
        json.dump(new_train_dataset, outf, indent=4)

    print(f"\nWriting dev data to: {output_dev_drop_json}")
    with open(output_dev_drop_json, 'w') as outf:
        json.dump(new_dev_dataset, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    main(args)
