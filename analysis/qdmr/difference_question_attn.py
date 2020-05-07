import os
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from semqa.utils.qdmr_utils import read_drop_dataset, convert_nestedexpr_to_tuple, Node, node_from_dict

from datasets.drop import constants

DIFF_PREDICATES = ["year_difference_two_events"] #, "passagenumber_difference"]


def get_cog(attention: List[int]):
    cog = sum([i * x for (i, x) in enumerate(attention)])
    # cog = max([i if x == 1 else 0 for (i, x) in enumerate(attention)])
    return cog


def convert_year_diff_two_events(qid, question, program_node: Node, count_dict):
    question = question.lower()
    relevant = False
    nested_expr = program_node.get_nested_expression()
    nested_expr_tuple = convert_nestedexpr_to_tuple(nested_expr)
    if nested_expr_tuple == ('year_difference_two_events', 'select_passage', 'select_passage'):
        count_dict["year-two-diff"] = count_dict.get("year-two-diff", 0) + 1
        select1_node = program_node.children[0]
        select2_node = program_node.children[1]
        qattn1 = select1_node.supervision["question_attention_supervision"]
        qattn2 = select2_node.supervision["question_attention_supervision"]
        cog1 = get_cog(qattn1)
        cog2 = get_cog(qattn2)
        if any([x in question for x in ["how many years passed",
                                        "how many years were between",
                                        "how many years was it"]]):
            count_dict["years-passed-between"] = count_dict.get("years-passed-between", 0) + 1
            # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
            if cog1 < cog2:
                # Switch select-node order in this case
                program_node.children = []
                program_node.add_child(select2_node)
                program_node.add_child(select1_node)
                count_dict["switched"] = count_dict.get("switched", 0) + 1
        elif "after" in question:
            count_dict["year-two-diff-after"] = count_dict.get("year-two-diff-after", 0) + 1
            if question[0:3] == "how":
                # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
                if cog1 < cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
            else:
                # We want cog1 < cog2, i.e., the first select1 to select the event mentioned earlier
                if cog1 > cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
        elif "how many years before" in question:
            count_dict["year-two-diff-before"] = count_dict.get("year-two-diff-before", 0) + 1
            if question[0:3] == "how":
                # We want cog1 <>> cog2, i.e., the first select1 to select the event mentioned earlier
                if cog1 > cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
            else:
                # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
                if cog1 < cog2:
                    # Switch select-node order in this case
                    program_node.children = []
                    program_node.add_child(select2_node)
                    program_node.add_child(select1_node)
                    count_dict["switched"] = count_dict.get("switched", 0) + 1
        else:
            count_dict["remaining"] = count_dict.get("remaining", 0) + 1
            # We want cog1 > cog2, i.e., the first select1 to select the event mentioned later
            if cog1 < cog2:
                # Switch select-node order in this case
                program_node.children = []
                program_node.add_child(select2_node)
                program_node.add_child(select1_node)
                count_dict["switched"] = count_dict.get("switched", 0) + 1
    return program_node


def convert_passage_diff(qid, question, program_node: Node, count_dict):
    question = question.lower()
    nested_expr = program_node.get_nested_expression()
    nested_expr_tuple = convert_nestedexpr_to_tuple(nested_expr)
    if "passagenumber_difference" in nested_expr_tuple:
        # print(f"{qid}\n{question}")
        # print(f"{program_node.get_nested_expression_with_strings()}")
        # print()
        count_dict["pass-diff"] = count_dict.get("pass-diff", 0) + 1


def analyze_difference_attn(dataset: Dict):
    """Analyze question-attention superivision/select-argument order in difference questions."""
    total_ques, total_relevant_ques = 0, 0
    count_dict = {}

    for passage_id, passage_info in dataset.items():
        qas = passage_info[constants.qa_pairs]
        total_ques += len(qas)
        for qa in qas:
            program_supervision = qa[constants.program_supervision]
            if not program_supervision:
                continue
            qid = qa[constants.query_id]
            question = qa[constants.question]
            passage = passage_info[constants.passage]
            program_node: Node = node_from_dict(program_supervision)
            program_node: Node = convert_year_diff_two_events(qid, question, program_node, count_dict)
            convert_passage_diff(qid, question, program_node, count_dict)
            qa[constants.program_supervision] = program_node.to_dict()



            # nested_expr = program_node.get_nested_expression()
            # nested_expr_tuple = convert_nestedexpr_to_tuple(nested_expr)
            # if any([x in nested_expr_tuple for x in DIFF_PREDICATES]):
            #     print(question)
            #     print(nested_expr_tuple)
            #     print(json.dumps(program_supervision, indent=2))
            #     if "after" in question:
            #         count_dict["after"] = count_dict.get("after", 0) + 1
            #     print()
            #     total_relevant_ques += 1

    print(f"Number of total ques: {total_ques}\nNumber of relevat ques: {total_relevant_ques}")
    print(count_dict)

    return dataset


def main(args):
    drop_data = read_drop_dataset(args.drop_json)
    dataset = analyze_difference_attn(drop_data)

    if args.out_json:
        output_json = args.out_json
        output_dir = os.path.split(output_json)[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"Writing drop data: {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(dataset, outf, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_json")
    parser.add_argument("--out_json")
    args = parser.parse_args()

    main(args)

