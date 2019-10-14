import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from datasets.drop import constants


""" Merge my heuristically annotated data with the full dataset.
    
    Use my question_answer pair for query_ids that are shared between my dataset and the full dataset.
"""


def count_supervision_types(passage_dict):
    supervision_keys = [
        constants.program_supervised,
        constants.qattn_supervised,
        constants.exection_supervised,
        constants.strongly_supervised,
    ]
    supervision_dict = defaultdict(int)
    for _, pinfo in passage_dict.items():
        qa_pairs = pinfo[constants.qa_pairs]
        for qa in qa_pairs:
            for key in supervision_keys:
                if key in qa and qa[key] is True:
                    supervision_dict[key] += 1

    return supervision_dict


def read_dataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset



def convert_mydata_into_qadict(mydata):
    # Dict from passage_id : Dict[qid: qa_pair]
    passageid2qid2qapair = {}

    for passage_id, passage_info in mydata.items():
        qa_pairs = passage_info[constants.qa_pairs]
        qid2qapair = {}
        for qa_pair in qa_pairs:
            qid = qa_pair[constants.query_id]
            qid2qapair[qid] = qa_pair
        passageid2qid2qapair[passage_id] = qid2qapair

    return passageid2qid2qapair



def mergeDatasets(mydata_json: str, fulldata_json: str, output_json: str) -> Dict[str, Dict]:
    """ Merge MyData subset of the full DROP dataset w/ the full dataset.
        Following conditions need to be handled
        1. QA only in FullData -- add as it is
        2. QA only in MyData -- add as it is (these would be DateComp and NumComp augmentations)
        3. QA in both -- add version from MyData (it would probably have auxiliary supervision)
    """

    mydata = read_dataset(mydata_json)
    fulldata = read_dataset(fulldata_json)

    mydata_passage_ids = set(mydata.keys())
    fulldata_passage_ids = set(fulldata.keys())

    union_passage_ids = mydata_passage_ids.union(fulldata_passage_ids)

    merged_data = {}

    num_output_qas = 0
    num_qas_1, num_qas_2 = 0, 0

    num_my_qa = 0
    num_full_qa = 0
    num_merged_qa = 0

    for passage_id in union_passage_ids:
        # Passage in Full data and not in MyData
        if passage_id in fulldata and passage_id not in mydata:
            merged_data[passage_id] = fulldata[passage_id]
            num_qas = len(fulldata[passage_id][constants.qa_pairs])
            num_full_qa += num_qas
            num_merged_qa += num_qas

        # Passage in MyData but not in FullData
        elif passage_id in mydata and passage_id not in fulldata:
            merged_data[passage_id] = mydata[passage_id]
            num_qas = len(mydata[passage_id][constants.qa_pairs])
            num_my_qa += num_qas
            num_merged_qa += num_qas

        elif passage_id in fulldata and passage_id in mydata:
            full_pinfo = fulldata[passage_id]
            my_pinfo = mydata[passage_id]

            new_pinfo = copy.copy(full_pinfo)

            full_qa_pairs = full_pinfo[constants.qa_pairs]
            full_qid2qapair = {qapair[constants.query_id]:qapair for qapair in full_qa_pairs}
            full_qids = set(full_qid2qapair.keys())
            num_full_qa += len(full_qid2qapair)

            my_qa_pairs = my_pinfo[constants.qa_pairs]
            my_qid2qapair = {qapair[constants.query_id]:qapair for qapair in my_qa_pairs}
            my_qids = set(my_qid2qapair.keys())
            num_my_qa += len(my_qid2qapair)

            new_qapairs = []
            union_qa_ids = full_qids.union(my_qids)
            for qid in union_qa_ids:
                if qid in my_qid2qapair:
                    new_qapairs.append(my_qid2qapair[qid])
                elif qid in full_qid2qapair:
                    new_qapairs.append(full_qid2qapair[qid])
                else:
                    raise RuntimeError

            new_pinfo[constants.qa_pairs] = new_qapairs
            num_merged_qa += len(new_qapairs)

            merged_data[passage_id] = new_pinfo
        else:
            raise RuntimeError

    supervision_dict = count_supervision_types(merged_data)
    print()
    print(f"Number of passages MY: {len(mydata_passage_ids)}\nNumber of questions MY: {num_my_qa}")
    print(f"Number of passages FULL: {len(fulldata_passage_ids)}\nNumber of questions FULL: {num_full_qa}")
    print(f"Number of merged passages: {len(merged_data)}\nNumber of merged questions: {num_merged_qa}")
    print(f"SupervisionDict: {supervision_dict}")
    print()

    print("Writing merged data to : {}".format(output_json))
    with open(output_json, "w") as f:
        json.dump(merged_data, f, indent=4)
    print("Written!\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mydata_dir")
    parser.add_argument("--fulldata_dir")
    parser.add_argument("--merged_dir")
    args = parser.parse_args()

    mydata_dir = args.mydata_dir
    fulldata_dir = args.fulldata_dir
    merged_dir = args.merged_dir

    # Raise error if directory already exists
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    FILES_TO_MERGE = ["drop_dataset_train.json", "drop_dataset_dev.json"]

    for filename in FILES_TO_MERGE:
        print(filename)
        mydata_json = os.path.join(mydata_dir, filename)
        fulldata_json = os.path.join(fulldata_dir, filename)
        merged_json = os.path.join(merged_dir, filename)
        print(f"MyData Json: {mydata_json}")
        print(f"FullData Json: {fulldata_json}")
        print(f"Merged Json: {merged_json}")

        mergeDatasets(mydata_json=mydata_json, fulldata_json=fulldata_json, output_json=merged_json)
