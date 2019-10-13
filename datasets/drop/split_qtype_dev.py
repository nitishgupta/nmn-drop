from typing import List
import os
import json
import argparse
from collections import defaultdict
import random
from utils import util

random.seed(100)


""" Let's say the full dataset's dev set paragraphs are split in to P_dev and P_test, then this script splits the 
    individual qtype datasets' dev set into mydev and mytest so that mytest will only contain paras from P_test.  
    
    Each directory inside args.root_qtype_datasets_dir is considered an individual qtype-dataset.
"""


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def splitDataset(dataset, test_paraids: List[str]):
    """Split dataset based on given test para ids"""

    total_num_passages = len(dataset)

    dev_dataset = {}
    test_dataset = {}

    for passage_idx, passage_info in dataset.items():
        if passage_idx in test_paraids:
            test_dataset[passage_idx] = passage_info
        else:
            dev_dataset[passage_idx] = passage_info

    print(f"TotalNumPassages: {total_num_passages}")
    print(f"Size of dev: {len(dev_dataset)} Size of test: {len(test_dataset)}")

    return dev_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fulldataset_dir")
    parser.add_argument("--root_qtype_datasets_dir")
    args = parser.parse_args()

    fulldataset_dir = args.fulldataset_dir
    root_qtype_datasets_dir = args.root_qtype_datasets_dir

    QTYPE_DATASET = ["count", "datecomp_full", "how_many_yards_was", "numcomp_full", "who_arg", "year_diff"]

    fulldataset_test_json = os.path.join(fulldataset_dir, "drop_dataset_mytest.json")

    fulldataset_testset = readDataset(fulldataset_test_json)
    test_para_ids = list(fulldataset_testset.keys())

    for qtype_dataset in QTYPE_DATASET:
        print("Splitting {}".format(qtype_dataset))
        dataset_dir = os.path.join(root_qtype_datasets_dir, qtype_dataset)
        input_devfp = os.path.join(dataset_dir, "drop_dataset_dev.json")
        output_devfp = os.path.join(dataset_dir, "drop_dataset_mydev.json")
        output_testfp = os.path.join(dataset_dir, "drop_dataset_mytest.json")

        orig_devset = readDataset(input_devfp)
        split_devset, split_testset = splitDataset(orig_devset, test_para_ids)

        with open(output_devfp, "w") as f:
            json.dump(split_devset, f, indent=4)

        with open(output_testfp, "w") as f:
            json.dump(split_testset, f, indent=4)
