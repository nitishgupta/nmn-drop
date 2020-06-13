from typing import List
from collections import defaultdict
import random
import argparse

from datasets.drop import constants
from semqa.utils.qdmr_utils import Node, convert_nestedexpr_to_tuple, node_from_dict, read_drop_dataset, \
    nested_expression_to_lisp


# def get_operators(nested_expression):
#     function_names = set()
#     operator_template = []
#     for i, argument in enumerate(nested_expression):
#         if i == 0:
#             function_names.add(argument)
#             operator_template.append(argument)
#         else:
#             if isinstance(argument, list):
#                 func_set, op_template = get_operators(argument)
#                 function_names.update(func_set)
#                 operator_template.extend(op_template)
#
#     return function_names, operator_template


def read_qdmr(drop_dataset):
    """ This data is processed using parse_dataset.parse_qdmr and keys in this json can be glanced at from there.

    The nested_expression in the data makes life easier since functions are already normalized to their min/max, add/sub
    identifier.

    This function mainly reads relevant
    """
    total_ques = 0
    total_super = 0

    qid2ques = {}
    qid2optemplate = {}
    function2qids = defaultdict(list)
    operatortemplate2count = defaultdict(int)
    lisp2count = defaultdict(int)
    operatortemplate2qids = defaultdict(list)
    qid2nestedexp = {}

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
            operator_template = convert_nestedexpr_to_tuple(nested_expr)
            lisp = nested_expression_to_lisp(nested_expr)

            qid2ques[query_id] = question
            qid2optemplate[query_id] = operator_template
            qid2nestedexp[query_id] = program_node.get_nested_expression_with_strings()
            operatortemplate2count[operator_template] += 1
            operatortemplate2qids[operator_template].append(query_id)
            lisp2count[lisp] += 1

    print("Total questions: {}  Total program abstractions: {}".format(total_ques, len(operatortemplate2count)))
    return qid2ques, operatortemplate2count, operatortemplate2qids, qid2nestedexp, lisp2count


def train_dev_stats(train_qid2ques, train_optemplate2count, train_optemplate2qids, train_lisp2count,
                    dev_qid2ques, dev_optemplate2count=None, dev_optemplate2qids=None, dev_lisp2count=None):
    train_templates = set(train_optemplate2count.keys())
    print("Train number of program templates: {}\n".format(len(train_templates)))

    count2numtemplates = defaultdict(int)
    for template, count in train_optemplate2count.items():
        count2numtemplates[count] += 1

    count2numtemplates_sorted = sorted(count2numtemplates.items(), key=lambda x: x[0], reverse=True)
    print("Train count_of_template vs. number of templates with that count -- ")
    print("{}\n".format(count2numtemplates_sorted))


    if dev_optemplate2count is not None:
        dev_templates = set(dev_optemplate2count.keys())
        print("Dev number of program templates: {}".format(len(dev_templates)))
        train_dev_common = train_templates.intersection(dev_templates)
        dev_extra_templates = dev_templates.difference(train_templates)
        print("Train / Dev common program templates (this disregards arguments): {}".format(len(train_dev_common)))
        print("Dev extra abstract program templates: {}".format(len(dev_extra_templates)))
        print("Number of questions for each of the new dev-templates: {}".format(
            [dev_optemplate2count[x] for x in dev_extra_templates]))

    print("\nTrain templates")
    lisp2count_sorted = sorted(train_lisp2count.items(), key=lambda x: x[1], reverse=True)
    for i in range(0, 15):
        lisp, count = lisp2count_sorted[i]
        print("{} {}".format(lisp.upper(), count))

    print("\nDev templates")
    lisp2count_sorted = sorted(dev_lisp2count.items(), key=lambda x: x[1], reverse=True)
    for i in range(0, 15):
        lisp, count = lisp2count_sorted[i]
        print("{} {}".format(lisp.upper(), count))


def write_example_programs_tsv(output_tsv_path, qid2ques, qid2nestedexp, func2qids):
    """Write example (question, program) in TSV for Google Sheets.

    To ensure diversity, we first sample 10 questions for each function type.
    """
    print("Writing examples to TSV: {}".format(output_tsv_path))
    qid_ques_programs = []
    for func, qids in func2qids.items():
        print(func)
        random.shuffle(qids)
        for i in range(0, min(20, len(qids))):
            qid = qids[i]
            qid_ques_programs.append((func, qid, qid2ques[qid], qid2nestedexp[qid]))

    print("Total examples written: {}".format(len(qid_ques_programs)))

    with open(output_tsv_path, 'w') as outf:
        outf.write(f"Function\tQueryID\tQuestion\tProgram\n")
        # random.shuffle(qid_programs)
        for (func, qid, ques, program) in qid_ques_programs:
            out_str = f"{func}\t{qid}\t{ques}\t{program}\n"
            outf.write(out_str)


def main(args):
    # train_drop_json = "/shared/nitishg/data/drop-w-qdmr/drop_wqdmr_programs/drop_dataset_train.json"
    # dev_drop_json = "/shared/nitishg/data/drop-w-qdmr/drop_wqdmr_programs/drop_dataset_dev.json"
    train_drop_json = args.train_drop_json
    dev_drop_json = args.dev_drop_json

    train_drop = read_drop_dataset(train_drop_json)
    dev_drop = read_drop_dataset(dev_drop_json)

    train_qid2ques, train_optemplate2count, train_optemplate2qids, train_qid2nestedexp, \
        train_lisp2count = read_qdmr(drop_dataset=train_drop)

    dev_qid2ques, dev_optemplate2count, dev_optemplate2qids, dev_qid2nestedexp, \
        dev_lisp2count = read_qdmr(drop_dataset=dev_drop)

    train_dev_stats(train_qid2ques, train_optemplate2count, train_optemplate2qids, train_lisp2count,
                    dev_qid2ques, dev_optemplate2count, dev_optemplate2qids, dev_lisp2count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_drop_json", required=None)
    parser.add_argument("--dev_drop_json", default=None)
    args = parser.parse_args()

    main(args)
