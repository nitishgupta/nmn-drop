from typing import List, Tuple, Dict, Union
import os
import json
import copy
import argparse
from collections import defaultdict

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp
from datasets.drop import constants


"""
Make a question-based compositional split; move all questions from train/dev that contain test-modules in the test data 
and move other test questions to train/dev split. This would mean that some test passages would now occur in train/dev
and vice-versa, hence the name "question-based compositional split".

We would have to decide how to divide the `train` questions from the test set into train/dev splits.
Current decision, move all to train  
"""

def dataset_stats(dataset):
    nump = len(dataset)
    numq = sum([len(pinfo[constants.qa_pairs]) for _, pinfo in dataset.items()])
    return nump, numq


def get_composiitonal_split(train_dataset: Dict, dev_dataset: Dict, test_dataset: Dict, test_modules: List[str]):
    """Move test_template questions from train/dev into test. Move other template questions from test to train

    Parameters:
    ----------
    keep: `bool`
        If True, keep question iff one of the modules from the list appears in the question-program
        If False, question shouldn't contain any of the modules mentioned
    modules: `List[str]`
        List of module names that govern the pruning.
    """

    print("\nPreparing compositional-split ... ")
    print("\nTest modules:")
    print("{}".format("\n".join(test_modules)))
    print()

    print("Before compositional-split -- ")
    print("Train dataset: {}".format(dataset_stats(train_dataset)))
    print("Dev dataset: {}".format(dataset_stats(dev_dataset)))
    print("Test dataset: {}".format(dataset_stats(test_dataset)))

    test_templates = set()
    train_paras_to_remove = []
    for pid, passage_info in train_dataset.items():
        pruned_train_qas = []
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                progam_node = node_from_dict(qa[constants.program_supervision])
                program_lisp = nested_expression_to_lisp(progam_node.get_nested_expression())
                if any([x in program_lisp for x in test_modules]):
                    test_templates.add(program_lisp)
                    if pid not in test_dataset:
                        test_dataset[pid] = copy.deepcopy(passage_info)
                        # Empty questions for this passage in test data
                        test_dataset[pid][constants.qa_pairs] = []
                    test_dataset[pid][constants.qa_pairs].append(qa)
                else:
                    pruned_train_qas.append(qa)
            else:
                pruned_train_qas.append(qa)
        # If all questions from a passage have moved to test, remove this paragraph
        if not pruned_train_qas:
            train_paras_to_remove.append(pid)
        passage_info[constants.qa_pairs] = pruned_train_qas

    for pid in train_paras_to_remove:
        train_dataset.pop(pid)

    dev_paras_to_remove = []
    for pid, passage_info in dev_dataset.items():
        pruned_qas = []
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                progam_node = node_from_dict(qa[constants.program_supervision])
                program_lisp = nested_expression_to_lisp(progam_node.get_nested_expression())
                if any([x in program_lisp for x in test_modules]):
                    test_templates.add(program_lisp)
                    if pid not in test_dataset:
                        test_dataset[pid] = copy.deepcopy(passage_info)
                        # Empty questions for this passage in test data
                        test_dataset[pid][constants.qa_pairs] = []
                    test_dataset[pid][constants.qa_pairs].append(qa)
                else:
                    pruned_qas.append(qa)
            else:
                pruned_qas.append(qa)
        if not pruned_qas:
            dev_paras_to_remove.append(pid)
        passage_info[constants.qa_pairs] = pruned_qas

    for pid in dev_paras_to_remove:
        dev_dataset.pop(pid)

    test_paras_to_remove = []
    for pid, passage_info in test_dataset.items():
        pruned_test_questions = []
        for qa in passage_info[constants.qa_pairs]:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                progam_node = node_from_dict(qa[constants.program_supervision])
                program_lisp = nested_expression_to_lisp(progam_node.get_nested_expression())
                if any([x in program_lisp for x in test_modules]):
                    test_templates.add(program_lisp)
                    pruned_test_questions.append(qa)
                else:
                    if pid not in train_dataset:
                        train_dataset[pid] = copy.deepcopy(passage_info)
                        # Empty questions for this passage in test data
                        train_dataset[pid][constants.qa_pairs] = []
                    train_dataset[pid][constants.qa_pairs].append(qa)
            else:
                # If program supervision not available, move to train
                if pid not in train_dataset:
                    train_dataset[pid] = copy.deepcopy(passage_info)
                    # Empty questions for this passage in test data
                    test_templates[pid][constants.qa_pairs] = []
                train_dataset[pid][constants.qa_pairs].append(qa)

        if not pruned_test_questions:
            test_paras_to_remove.append(pid)
        passage_info[constants.qa_pairs] = pruned_test_questions

    for pid in test_paras_to_remove:
        test_dataset.pop(pid)

    test_templates = list(test_templates)
    print("\nTest templates:")
    print("{}".format("\n".join(test_templates)))

    print("\nAfter compositional-split -- ")
    print("Train dataset: {}".format(dataset_stats(train_dataset)))
    print("Dev dataset: {}".format(dataset_stats(dev_dataset)))
    print("Test dataset: {}".format(dataset_stats(test_dataset)))

    return train_dataset, dev_dataset, test_dataset


def merge_datasets(train_dataset, dev_dataset):
    merged_dataset = {}
    merged_dataset.update(train_dataset)
    merged_dataset.update(dev_dataset)
    return merged_dataset


def main(args):
    input_dir = args.input_dir

    train_drop_json = os.path.join(input_dir, "drop_dataset_train.json")
    dev_drop_json = os.path.join(input_dir, "drop_dataset_dev.json")
    test_drop_json = os.path.join(input_dir, "drop_dataset_test.json")

    train_drop = read_drop_dataset(train_drop_json)
    dev_drop = read_drop_dataset(dev_drop_json)
    test_drop = read_drop_dataset(test_drop_json)

    test_modules = ["select_min_num",
                    "compare_date_lt",
                    "compare_num_lt"]

    trdata, devdata, testdata = get_composiitonal_split(train_drop, dev_drop, test_drop, test_modules=test_modules)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print("\nWriting outputs to: {}".format(output_dir))
    with open(os.path.join(output_dir, "drop_dataset_train.json"), 'w') as outf:
        json.dump(trdata, outf, indent=4)

    with open(os.path.join(output_dir, "drop_dataset_dev.json"), 'w') as outf:
        json.dump(devdata, outf, indent=4)

    with open(os.path.join(output_dir, "drop_dataset_test.json"), 'w') as outf:
        json.dump(testdata, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    main(args)
