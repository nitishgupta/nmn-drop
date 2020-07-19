from typing import List, Dict
import os
from collections import defaultdict
import random
import argparse
from dataclasses import dataclass

from datasets.drop import constants
from semqa.utils.qdmr_utils import Node, convert_nestedexpr_to_tuple, node_from_dict, read_drop_dataset, \
    nested_expression_to_lisp, convert_answer

@dataclass
class Example:
    question: str
    passage: str
    lisp: str
    program: str
    answer: Dict
    program_node: Node



def is_td_event(string_arg: str):
    string_arg = string_arg.lower()
    td_event = False
    if (any(x in string_arg for x in ["touchdown", "field goal", "interceptions", "passes", "td", "catches",
                                      "punts", "runs", "fumbles", "reception", "score", "scoring", "throw",
                                      "intercept"])):
        td_event = True
    return td_event


def is_football_agg(string_arg: str):
    string_arg = string_arg.lower()
    ft_event = False
    if (any(x in string_arg for x in ["games", "wins", "points", "losses", "win", "lose"])):
        ft_event = True
    return ft_event


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

    qid2example = {}
    num_count_ques = 0
    num_numans_ques = 0
    ans_in_passagenums = 0
    num_td_ques = 0
    ans_in_passage_not_td = 0
    num_ft_ques = 0
    td_and_ft = 0
    not_td_or_ft = 0

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

            if lisp != "(aggregate_count select_passage)":
                continue
            num_count_ques += 1
            string_arg = program_node.children[0].string_arg
            passage_nums = passage_info[constants.passage_num_normalized_values]
            answer_type, answer_texts = convert_answer(qa[constants.answer])

            if answer_type == "number":
                num_numans_ques += 1
                num_ans = float(answer_texts[0])
                ans_in_passagenums += 1 if num_ans in passage_nums else 0

            num_td_ques += 1 if is_td_event(string_arg) else 0
            num_ft_ques += 1 if is_football_agg(string_arg) else 0
            td_and_ft += 1 if is_football_agg(string_arg) and is_td_event(string_arg) else 0
            # not_td_or_ft += 1 if not (is_football_agg(string_arg) or is_td_event(string_arg)) else 0

            if not (is_football_agg(string_arg) or is_td_event(string_arg)):
                passage = passage_info[constants.passage]
                if not any([x in passage for x in ["touchdown", "field goal"]]):
                    not_td_or_ft += 1
                    print(question)
                    print(passage)
                    print(answer_texts)
                    print()

            lisp2qids[lisp].append(query_id)
            lisp2count[lisp] += 1

            example = Example(question=question, passage=passage_info[constants.passage], lisp=lisp,
                              program=program_node.get_nested_expression_with_strings(), answer=qa[constants.answer],
                              program_node=program_node)
            qid2example[query_id] = example

    print("Total questions: {}  Total program abstractions: {}".format(total_ques, len(lisp2count)))

    print(f"Count ques: {num_count_ques}  num_numans_ques: {num_numans_ques}  ans_in_passagenums: {ans_in_passagenums} "
          f" num_td_ques: {num_td_ques}  num_ft_ques:{num_ft_ques}  td_and_ft: {td_and_ft} not_td_or_ft: {not_td_or_ft}")

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

    train_drop = read_drop_dataset(train_drop_json)
    dev_drop = read_drop_dataset(dev_drop_json)

    train_lisp2count, train_lisp2qids, train_qid2example = read_qdmr(drop_dataset=train_drop)

    # dev_qid2ques, dev_lisp2count, dev_lisp2qids, dev_qid2nestedexp = read_qdmr(drop_dataset=dev_drop)

    # relevant_lisps = {
    #     "count_select": "(aggregate_count select_passage)",
    #     "passage_diff_lisp": "(passagenumber_difference (select_num select_passage) (select_num select_passage))",
    #     "select_num": "(select_num select_passage)",
    # }
    #
    # for lisp_name, lisp in relevant_lisps.items():
    #     lisp_output_tsv_path = os.path.join(input_dir, lisp_name + ".tsv")
    #     print(lisp_output_tsv_path)
    #     write_example_programs_tsv(lisp_output_tsv_path=lisp_output_tsv_path,
    #                                lisp=lisp,
    #                                lisp2qids=train_lisp2qids,
    #                                qid2example=train_qid2example)
    #     print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    args = parser.parse_args()

    main(args)