from typing import List, Tuple, Dict
import os
import json
import argparse
from collections import defaultdict
import random
from utils import util
import datasets.drop.constants as dropconstants

random.seed(100)


""" Let's say the full dataset's dev set paragraphs are split in to P_dev and P_test, then this script splits the 
    individual qtype datasets' dev set into mydev and mytest so that mytest will only contain paras from P_test.  
    
    Each directory inside args.root_qtype_datasets_dir is considered an individual qtype-dataset.
"""


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


def get_num_passage_ques(dataset):
    total_num_passages = len(dataset)
    total_num_qas = 0
    for passage_idx, passage_info in dataset.items():
        total_num_qas += len(passage_info[dropconstants.qa_pairs])

    return total_num_passages, total_num_qas


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


def get_split_paragraphids(fulldataset, qtype2dataset, split_ratio) -> Tuple[List[str], List[str]]:
    """Split dataset based on given test para ids"""

    passage_idxs = list(fulldataset.keys())
    total_num_passages = len(passage_idxs)
    total_num_ques = sum([len(fulldataset[pid][dropconstants.qa_pairs]) for pid in fulldataset])
    num_dev_paras = int(split_ratio * total_num_passages)

    # Try multiple splits, since ratio of questions might vary even if paras are split according to ratio
    num_of_splits_to_try = 5000

    best_diff_split_actual_ratio = 100.0
    best_ratios = []
    best_mydev_para_ids = None
    best_mytest_para_ids = None
    for trynum in range(num_of_splits_to_try):
        # print("Try number: {}".format(trynum))
        random.shuffle(passage_idxs)
        mydev_passage_idxs = passage_idxs[0:num_dev_paras]
        mytest_passage_idxs = passage_idxs[num_dev_paras:]
        mydev_passage_idxs = set(mydev_passage_idxs)
        mytest_passage_idxs = set(mytest_passage_idxs)
        assert len(mydev_passage_idxs.intersection(mytest_passage_idxs)) == 0

        ratio_diffs = []
        ratios = []
        for qtype, qtypedataset in qtype2dataset.items():
            qtype_totalnum_qas = sum(len(qtypedataset[pid][dropconstants.qa_pairs]) for pid in qtypedataset)
            qtype_mydevnum_qas = 0
            for pid in mydev_passage_idxs:
                if pid in qtypedataset:
                    qtype_mydevnum_qas += len(qtypedataset[pid][dropconstants.qa_pairs])

            qtype_mydev_ratio = float(qtype_mydevnum_qas) / float(qtype_totalnum_qas)
            ratios.append(qtype_mydev_ratio)
            ratio_diff = abs(qtype_mydev_ratio - split_ratio)
            ratio_diffs.append(ratio_diff)

        max_ratio_diff = max(ratio_diffs)
        if max_ratio_diff < best_diff_split_actual_ratio:
            best_diff_split_actual_ratio = max_ratio_diff
            best_ratios = ratios
            best_mydev_para_ids = mydev_passage_idxs
            best_mytest_para_ids = mytest_passage_idxs

    print("Best dev/test qa ratio diff: {}".format(best_diff_split_actual_ratio))
    print("Best ratios: {}".format(best_ratios))

    return set(best_mydev_para_ids), set(best_mytest_para_ids)


def make_qtype2dataset_map(qtype_datasets_rootdir, qtype_datasets, filename):
    # Qtype 2 Dataset map
    qtype2dataset = {}
    for qtype_dirname in qtype_datasets:
        data_json = os.path.join(qtype_datasets_rootdir, qtype_dirname, filename)
        qtype_data = readDataset(data_json)
        qtype2dataset[qtype_dirname] = qtype_data
        # num_passages_qtype, num_qas_qtype = get_num_passage_ques(qtype_data)
        # ratio_of_full_dev = float(num_qas_qtype) * 100.0 / num_qas_dev
        # print("Number in {}; passages: {}  QA:{} Ratio_dev: {}".format(
        #     qtype_dirname, num_passages_qtype, num_qas_qtype, ratio_of_full_dev))

    return qtype2dataset


def qtyperatio2fulldataset(full_dataset, qtype2dataset):
    num_passages_full, num_qas_full = get_num_passage_ques(full_dataset)
    for qtype, dataset in qtype2dataset.items():
        num_passages_qtype, num_qas_qtype = get_num_passage_ques(dataset)
        ratio_of_full_dev = float(num_qas_qtype) * 100.0 / num_qas_full
        print(
            "Number in {}; passages: {}  QA:{} Ratio_dev: {}".format(
                qtype, num_passages_qtype, num_qas_qtype, ratio_of_full_dev
            )
        )


def write_dataset(dataset, output_dir, output_filname):
    with open(os.path.join(output_dir, output_filname), "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fulldataset_dir")
    parser.add_argument("--qtype_dir_name")
    parser.add_argument("--split_ratio", type=float)  # ratio of mydev to mytest
    args = parser.parse_args()

    # This is full merged dataset -- should contain drop_dataset_dev.json
    fulldataset_dir = args.fulldataset_dir

    # This directory should contain multiple qtype dirs, after merging which "fulldataset_dir" was created
    # Each qtype dir will contain a drop_dataset_dev.json (union of which is fulldataset_dir/drop_dataset_dev.json)
    qtype_datasets_rootdir = os.path.join(fulldataset_dir, args.qtype_dir_name)

    print("Different question type datasets:")
    qtype_datasets = os.listdir(qtype_datasets_rootdir)
    print(qtype_datasets)

    full_dev_json = os.path.join(fulldataset_dir, "drop_dataset_dev.json")
    fulldev_dataset = readDataset(full_dev_json)
    num_passages_dev, num_qas_dev = get_num_passage_ques(fulldev_dataset)
    print("Number of full Dev; passages: {}  QA:{}".format(num_passages_dev, num_qas_dev))

    # Qtype 2 Dataset map
    qtype2dataset = make_qtype2dataset_map(qtype_datasets_rootdir, qtype_datasets, "drop_dataset_dev.json")
    qtyperatio2fulldataset(fulldev_dataset, qtype2dataset)

    # Divide the full_dataset dev para in to mydev and mytest in the split_ratio
    # Also try to keep the same ratio of different qtypes in mydev and mytest
    mydev_paraids, mytest_paraids = get_split_paragraphids(fulldev_dataset, qtype2dataset, args.split_ratio)

    assert len(mydev_paraids.intersection(mytest_paraids)) == 0

    mydev_dataset, mytest_dataset = {}, {}
    for pid, pinfo in fulldev_dataset.items():
        if pid in mydev_paraids:
            mydev_dataset[pid] = pinfo
        elif pid in mytest_paraids:
            mytest_dataset[pid] = pinfo
        else:
            print("pid not in mydev or mytest: {}".format(pid))
            raise RuntimeError

    num_passages_mydev, num_qas_mydev = get_num_passage_ques(mydev_dataset)
    print("Number of My Dev; passages: {}  QA:{}".format(num_passages_mydev, num_qas_mydev))

    num_passages_mytest, num_qas_mytest = get_num_passage_ques(mytest_dataset)
    print("Number of My Test; passages: {}  QA:{}".format(num_passages_mytest, num_qas_mytest))

    qtype2mydevdata = {}
    qtype2mytestdata = {}
    for qtype, dataset in qtype2dataset.items():
        qtype_mydev_dataset = {}
        qtype_mytest_dataset = {}
        for pid, pinfo in dataset.items():
            if pid in mydev_paraids:
                qtype_mydev_dataset[pid] = pinfo
            elif pid in mytest_paraids:
                qtype_mytest_dataset[pid] = pinfo
            else:
                raise RuntimeError
        qtype2mydevdata[qtype] = qtype_mydev_dataset
        qtype2mytestdata[qtype] = qtype_mytest_dataset
    print("Qtype analysis for my dev")
    qtyperatio2fulldataset(mydev_dataset, qtype2mydevdata)
    print("Qtype analysis for my test")
    qtyperatio2fulldataset(mytest_dataset, qtype2mytestdata)

    write_dataset(mydev_dataset, fulldataset_dir, "drop_dataset_mydev.json")
    write_dataset(mytest_dataset, fulldataset_dir, "drop_dataset_mytest.json")

    for qtype, devdataset in qtype2mydevdata.items():
        write_dataset(devdataset, os.path.join(qtype_datasets_rootdir, qtype), "drop_dataset_mydev.json")

    for qtype, testdataset in qtype2mytestdata.items():
        write_dataset(testdataset, os.path.join(qtype_datasets_rootdir, qtype), "drop_dataset_mytest.json")
