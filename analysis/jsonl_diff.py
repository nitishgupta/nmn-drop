from typing import List, Dict
import json
from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, nested_expression_to_lisp

old_json = "/shared/nitishg/data/drop-w-qdmr/qdmr-filter-post-v6/drop_dataset_train.json"
new_json = "/shared/nitishg/data/drop-w-qdmr/qdmr-filter-v1/drop_dataset_train.json"


old_dataset = read_drop_dataset(old_json)
new_dataset = read_drop_dataset(new_json)

def get_qid2question(dataset):
    qid2question = {}
    for passage_id, passage_info in dataset.items():
        qas = passage_info[constants.qa_pairs]
        for qa in qas:
            qid2question[qa[constants.query_id]] = qa[constants.question]
    return qid2question

def relevant_qids(dataset, lisp):
    qids = set()
    qid2question = {}
    for passage_id, passage_info in dataset.items():
        qas = passage_info[constants.qa_pairs]
        for qa in qas:
            if constants.program_supervision in qa and qa[constants.program_supervision]:
                node = node_from_dict(qa[constants.program_supervision])
                qa_lisp = nested_expression_to_lisp(node.get_nested_expression())
                if qa_lisp == lisp:
                    qids.add(qa[constants.query_id])
                    qid2question[qa[constants.query_id]] = qa[constants.question]

    print("Lisp: {}".format(lisp))
    print("QIDs: {}".format(len(qids)))
    return qids


def print_diff(old_qids, new_qids, old_qid2question, new_qid2question):
    qid_diff = old_qids.difference(new_qids)
    for qid in qid_diff:
        print(old_qid2question[qid])



numgt = "(select_passagespan_answer (compare_num_gt select_passage select_passage))"
numlt = "(select_passagespan_answer (compare_num_lt select_passage select_passage))"

dategt = "(select_passagespan_answer (compare_date_gt select_passage select_passage))"
datelt = "(select_passagespan_answer (compare_date_lt select_passage select_passage))"

old_qid2question = get_qid2question(old_dataset)
new_qid2question = get_qid2question(new_dataset)

print("old dataset")
old_numgt = relevant_qids(old_dataset, numgt)
old_numlt = relevant_qids(old_dataset, numlt)
old_dategt = relevant_qids(old_dataset, dategt)
old_dategt = relevant_qids(old_dataset, datelt)

print("\nnew dataset")
new_numgt = relevant_qids(new_dataset, numgt)
new_numlt = relevant_qids(new_dataset, numlt)
new_dategt = relevant_qids(new_dataset, dategt)
new_dategt = relevant_qids(new_dataset, datelt)

print()

print_diff(old_numlt, new_numlt, old_qid2question, new_qid2question)



