import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from datasets.drop import constants


def count_supervision_types(passage_dict):
    supervision_dict = defaultdict(int)
    for _, pinfo in passage_dict.items():
        qa_pairs = pinfo[constants.qa_pairs]
        for qa in qa_pairs:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                supervision_dict["program_supervision"] += 1
            if constants.execution_supervised in qa and qa[constants.execution_supervised]:
                supervision_dict[constants.execution_supervised] += 1

    return supervision_dict


def mergeDatasets(input_json1: str, input_json2: str, output_json: str) -> None:
    """ Merge DROP datasets from two different files.
        First make a union list of passages.
            If a passage only appears in one - add as it is
            If occurs in both, merge questions inside
                Since question_answer is a dict; check for duplicacy by making a set of question_dicts first.
                If question occurs in both, input_json1 gets preference.
    """

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json1, "r") as f:
        dataset1 = json.load(f)

    with open(input_json2, "r") as f:
        dataset2 = json.load(f)

    passage_ids_1 = set(dataset1.keys())
    passage_ids_2 = set(dataset2.keys())

    union_passage_ids = passage_ids_1.union(passage_ids_2)

    num_passage_1 = len(passage_ids_1)
    num_passage_2 = len(passage_ids_2)
    num_merged_passages = len(union_passage_ids)

    output_passage_dict = {}
    num_output_qas = 0
    num_qas_1, num_qas_2 = 0, 0

    for passage_id in union_passage_ids:
        if passage_id in passage_ids_1 and passage_id in passage_ids_2:
            pinfo_1 = dataset1[passage_id]
            pinfo_2 = dataset2[passage_id]

            # Assumes that all passage-level parsing is same in both datasets, only qa_pairs differ
            passage_info = copy.copy(pinfo_1)

            # Dataset1 gets preference for QA if query_id is present in both
            qa_pairs_1 = pinfo_1[constants.qa_pairs]
            qid2qapair_1 = {qapair[constants.query_id]: qapair for qapair in qa_pairs_1}
            qids_1 = set(qid2qapair_1.keys())
            num_qas_1 += len(qids_1)

            qa_pairs_2 = pinfo_2[constants.qa_pairs]
            qid2qapair_2 = {qapair[constants.query_id]: qapair for qapair in qa_pairs_2}
            qids_2 = set(qid2qapair_2.keys())
            num_qas_2 += len(qids_2)

            new_qapairs = []
            union_qa_ids = qids_1.union(qids_2)
            for qid in union_qa_ids:
                if qid in qids_1:
                    new_qapairs.append(qid2qapair_1[qid])
                else:
                    new_qapairs.append(qid2qapair_2[qid])
            passage_info[constants.qa_pairs] = new_qapairs

        elif passage_id in passage_ids_1:
            passage_info = copy.deepcopy(dataset1[passage_id])
            num_qas_1 += len(passage_info[constants.qa_pairs])
        elif passage_id in passage_ids_2:
            passage_info = copy.deepcopy(dataset2[passage_id])
            num_qas_2 += len(passage_info[constants.qa_pairs])
        else:
            raise RuntimeError

        output_passage_dict[passage_id] = passage_info
        num_output_qas += len(passage_info[constants.qa_pairs])

    with open(output_json, "w") as outf:
        json.dump(output_passage_dict, outf, indent=4)

    supervision_dict = count_supervision_types(output_passage_dict)
    print()
    print(f"Number of passages 1: {num_passage_1}\nNumber of questions 1: {num_qas_1}")
    print(f"Number of passages 2: {num_passage_2}\nNumber of questions 2: {num_qas_2}")
    print(f"Number of merged passages: {num_merged_passages}\nNumber of merged questions: {num_output_qas}")
    print(f"SupervisionDict: {supervision_dict}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1")
    parser.add_argument("--dir2")
    parser.add_argument("--outputdir")
    args = parser.parse_args()

    dir1 = args.dir1
    dir2 = args.dir2
    outputdir = args.outputdir

    # Raise error if directory already exists
    os.makedirs(outputdir, exist_ok=True)

    FILES_TO_MERGE = ["drop_dataset_train.json", "drop_dataset_dev.json"]

    for filename in FILES_TO_MERGE:
        print(filename)
        file1 = os.path.join(dir1, filename)
        file2 = os.path.join(dir2, filename)
        outputfile = os.path.join(outputdir, filename)
        print(f"File1: {file1}")
        print(f"File2: {file2}")
        print(f"OutFile: {outputfile}")

        mergeDatasets(input_json1=file1, input_json2=file2, output_json=outputfile)
