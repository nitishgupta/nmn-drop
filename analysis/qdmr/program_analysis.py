from typing import List, Dict
import os
from collections import defaultdict
import random
import argparse
from dataclasses import dataclass

from datasets.drop import constants
from semqa.utils.qdmr_utils import Node, convert_nestedexpr_to_tuple, node_from_dict, read_drop_dataset, \
    nested_expression_to_lisp

@dataclass
class Example:
    question: str
    passage: str
    lisp: str
    program: str
    answer: Dict




def read_qdmr(drop_dataset):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    This function mainly reads relevant
    """
    total_ques = 0
    function2qids = defaultdict(list)
    lisp2count = defaultdict(int)
    lisp2qids = defaultdict(list)
    qid2nestedexp = {}

    qid2example = {}

    for passage_id, passage_info in drop_dataset.items():

        for qa in passage_info[constants.qa_pairs]:
            total_ques += 1
            question = qa[constants.question]
            query_id = qa[constants.query_id]
            program_supervision = qa.get(constants.program_supervision, None)
            if program_supervision is None:
                continue
            program_node: Node = node_from_dict(program_supervision)
            nested_expr = program_node.get_nested_expression()
            lisp = nested_expression_to_lisp(nested_expr)

            qid2nestedexp[query_id] = program_node.get_nested_expression_with_strings()
            lisp2qids[lisp].append(query_id)
            lisp2count[lisp] += 1

            example = Example(question=question, passage=passage_info[constants.passage], lisp=lisp,
                              program=program_node.get_nested_expression_with_strings(), answer=qa[constants.answer])
            qid2example[query_id] = example

    print("Total questions: {}  Total program abstractions: {}".format(total_ques, len(lisp2count)))
    return lisp2count, lisp2qids, qid2example


def write_example_programs_tsv(lisp_output_tsv_path, lisp, lisp2qids, qid2example):
    """Write example (lisp, question, program, answer) in TSV for Google Sheets.

    To ensure diversity, we first sample 10 questions for each function type.
    """

    print("Writing for : {}".format(lisp))
    qids = lisp2qids[lisp]
    random.shuffle(qids)

    with open(lisp_output_tsv_path, 'w') as outf:
        header = "Lisp\tqid\tQuestion\tProgram\tPassage\tAnswer\n"
        outf.write(header)

        for qid in qids:
            example: Example = qid2example[qid]
            question = example.question
            passage = example.passage
            answer = example.answer
            program = example.program
            outf.write(f"{lisp}\t{qid}\t{question}\t{program}\t{passage}\t{answer}\n")

def main(args):
    # train_drop_json = "/shared/nitishg/data/drop-w-qdmr/drop_wqdmr_programs/drop_dataset_train.json"
    # dev_drop_json = "/shared/nitishg/data/drop-w-qdmr/drop_wqdmr_programs/drop_dataset_dev.json"

    input_dir = args.input_dir

    train_drop_json = os.path.join(input_dir, "drop_dataset_train.json")
    dev_drop_json = os.path.join(input_dir, "drop_dataset_dev.json")

    train_lisp_examples_tsv_output = os.path.join(input_dir, "train_lisp.tsv")

    train_drop = read_drop_dataset(train_drop_json)
    # dev_drop = read_drop_dataset(dev_drop_json)

    train_lisp2count, train_lisp2qids, train_qid2example = read_qdmr(drop_dataset=train_drop)

    # dev_qid2ques, dev_lisp2count, dev_lisp2qids, dev_qid2nestedexp = read_qdmr(drop_dataset=dev_drop)

    relevant_lisps = {
        "count_select": "(aggregate_count select_passage)",
        "passage_diff_lisp": "(passagenumber_difference (select_num select_passage) (select_num select_passage))",
        "select_num": "(select_num select_passage)",
    }

    for lisp_name, lisp in relevant_lisps.items():
        lisp_output_tsv_path = os.path.join(input_dir, lisp_name + ".tsv")
        print(lisp_output_tsv_path)
        write_example_programs_tsv(lisp_output_tsv_path=lisp_output_tsv_path,
                                   lisp=lisp,
                                   lisp2qids=train_lisp2qids,
                                   qid2example=train_qid2example)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    args = parser.parse_args()

    main(args)
