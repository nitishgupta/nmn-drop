from typing import List, Tuple, Dict, Union
import os
import json
import copy
import argparse
from collections import defaultdict

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp
from datasets.drop import constants


"""
Pruned dataset to contain questions from only certain templates  
"""

def dataset_stats(dataset):
    nump = len(dataset)
    numq = sum([len(pinfo[constants.qa_pairs]) for _, pinfo in dataset.items()])
    return nump, numq


def get_templatepruned_dataset(dataset: Dict, templates_to_keep: List[str]):
    """Keep questions with program-template in templates_to_keep"""

    print("\nPruning dataset ... ")

    print("Before pruning -- ")
    print("Input dataset: {}".format(dataset_stats(dataset)))

    paras_to_remove = []
    for pid, passage_info in dataset.items():
        pruned_train_qas = []
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                progam_node = node_from_dict(qa[constants.program_supervision])
                program_lisp = nested_expression_to_lisp(progam_node.get_nested_expression())
                if program_lisp in templates_to_keep:
                    pruned_train_qas.append(qa)
        # If all questions from a passage have moved to test, remove this paragraph
        if not pruned_train_qas:
            paras_to_remove.append(pid)
        passage_info[constants.qa_pairs] = pruned_train_qas

    for pid in paras_to_remove:
        dataset.pop(pid)
    print("\nAfter pruning -- ")
    print("Output dataset: {}".format(dataset_stats(dataset)))
    return dataset


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    templates_to_keep = [
        # min/max FIND-NUM
        "(select_num (select_max_num select_passage))",
        "(select_num (select_min_num select_passage))",
        "(select_num (select_max_num (filter_passage select_passage)))",
        "(select_num (select_min_num (filter_passage select_passage)))",
        # min/max PROJECT
        "(select_passagespan_answer (project_passage (select_max_num select_passage)))",
        "(select_passagespan_answer (project_passage (select_min_num select_passage)))",
        "(select_passagespan_answer (project_passage (select_max_num (filter_passage select_passage))))",
        "(select_passagespan_answer (project_passage (select_min_num (filter_passage select_passage))))",
        # COUNT
        "(aggregate_count select_passage)",
        "(aggregate_count (filter_passage select_passage))",
    ]

    FILES_TO_MERGE = ["drop_dataset_train.json", "drop_dataset_dev.json", "drop_dataset_test.json",
                      "iclr21_filter_faithful.json"]

    for filename in FILES_TO_MERGE:
        print("\n\n")
        print(filename)
        input_json = os.path.join(input_dir, filename)
        output_json = os.path.join(output_dir, filename)

        if not os.path.exists(input_json):
            print(f"{input_json} does not exist. Skipping.")
            continue

        print(f"Input json: {input_json}")
        input_dataset = read_drop_dataset(input_json)
        output_dataset = get_templatepruned_dataset(input_dataset, templates_to_keep=templates_to_keep)
        print(f"Output json: {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(output_dataset, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    main(args)
