import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from semqa.utils.qdmr_utils import read_drop_dataset

from datasets.drop import constants


def mergeDatasets(dataset1: Dict, preferred_dataset: Dict) -> Dict:
    """ Merge DROP datasets from two different files.
        First make a union list of passages.
            If a passage only appears in one - add as it is
            If occurs in both, merge questions inside
                Since question_answer is a dict; check for duplicacy by making a set of question_dicts first.
    """

    passage_ids_1 = set(dataset1.keys())
    preferred_passage_ids = set(preferred_dataset.keys())

    union_passage_ids = passage_ids_1.union(preferred_passage_ids)

    num_passage_1 = len(passage_ids_1)
    num_passage_2 = len(preferred_passage_ids)
    num_merged_passages = len(union_passage_ids)

    num_full_qa, num_merged_qa, num_my_qa = 0, 0, 0

    merged_data = {}
    for passage_id in union_passage_ids:
        # Passage in Full data and not in MyData
        if passage_id in passage_ids_1 and passage_id not in preferred_passage_ids:
            merged_data[passage_id] = dataset1[passage_id]
            num_qas = len(dataset1[passage_id][constants.qa_pairs])
            num_full_qa += num_qas
            num_merged_qa += num_qas

        # Passage in MyData but not in FullData
        elif passage_id in preferred_passage_ids and passage_id not in passage_ids_1:
            merged_data[passage_id] = preferred_dataset[passage_id]
            num_qas = len(preferred_dataset[passage_id][constants.qa_pairs])
            num_my_qa += num_qas
            num_merged_qa += num_qas

        else:
            pinfo_1 = dataset1[passage_id]
            pinfo_prefer = preferred_dataset[passage_id]

            qas_1 = pinfo_1[constants.qa_pairs]
            qas_prefer = pinfo_prefer[constants.qa_pairs]

            final_pinfo: Dict = copy.deepcopy(pinfo_1)
            final_pinfo.update(pinfo_prefer)

            qas_1_qid2qapair = {qapair[constants.query_id]:qapair for qapair in qas_1}
            qas_1_qids = set(qas_1_qid2qapair.keys())
            num_full_qa += len(qas_1_qid2qapair)

            qas_prefer_qid2qapair = {qapair[constants.query_id]:qapair for qapair in qas_prefer}
            qas_prefer_qids = set(qas_prefer_qid2qapair.keys())
            num_my_qa += len(qas_prefer_qid2qapair)

            new_qapairs = []
            union_qa_ids = qas_1_qids.union(qas_prefer_qids)
            for qid in union_qa_ids:
                if qid in qas_prefer_qids:
                    new_qapairs.append(qas_prefer_qid2qapair[qid])
                else:
                    new_qapairs.append(qas_1_qid2qapair[qid])

            final_pinfo[constants.qa_pairs] = new_qapairs
            num_merged_qa += len(new_qapairs)

            merged_data[passage_id] = final_pinfo

    print()
    print(f"Number of passages 1: {num_passage_1}\nNumber of questions 1: {num_full_qa}")
    print(f"Number of passages 2: {num_passage_2}\nNumber of questions 2: {num_my_qa}")
    print(f"Number of merged passages: {num_merged_passages}\nNumber of merged questions: {num_merged_qa}")
    print()
    return merged_data


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    FILES_TO_MERGE = ["drop_dataset_train.json", "drop_dataset_dev.json"]

    for filename in FILES_TO_MERGE:
        print(filename)
        file1 = os.path.join(args.dir1, filename)
        file2 = os.path.join(args.dir_prefer, filename)
        output_json = os.path.join(args.output_dir, filename)
        print(f"File1: {file1}")
        print(f"File2: {file2}")
        print(f"OutFile: {output_json}")

        merged_data = mergeDatasets(dataset1=read_drop_dataset(file1), preferred_dataset=read_drop_dataset(file2))

        print(f"Writing merged data to : {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(merged_data, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1")
    parser.add_argument("--dir_prefer")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    main(args)

