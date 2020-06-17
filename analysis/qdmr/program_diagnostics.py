import os
import re
import json
import copy
import random
import argparse
import itertools
from typing import List, Tuple, Dict, Union, Callable

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    convert_answer, convert_nestedexpr_to_tuple, write_jsonl
from semqa.domain_languages.drop_language_v2 import Date

from datasets.drop import constants

random.seed(42)


def is_project_and_how(question: str, program_node: Node, program_lisp: str) -> bool:
    if program_node.predicate == "project_passage":
        if "how" in program_node.string_arg:
            match = True
            return match

    match = False
    for child in program_node.children:
        match = match or is_project_and_how(question, child, program_lisp)

    return match


def is_project_and_not_how(question: str, program_node: Node, program_lisp: str) -> bool:
    if program_node.predicate == "project_passage":
        if "how" not in program_node.string_arg:
            match = True
            return match

    match = False
    for child in program_node.children:
        match = match or is_project_and_not_how(question, child, program_lisp)

    return match



def is_filter_noquarterhalf(question: str, program_node: Node, program_lisp: str) -> bool:
    match = False
    if "filter_passage" in program_lisp:
        if not any([x in question for x in ["quarter?", "half?"]]):
            match = True
    return match



patterns_for_filternum = [
        "over [0-9]+",  # "over #NUM"
        "under [0-9]+",  # "over #NUM"
        "below [0-9]+",  # "over #NUM"
        "between [0-9]+",  # "over #NUM"
        "at least [0-9]+",  # "over #NUM"
        "atleast [0-9]+",  # "over #NUM"
        "at most [0-9]+",  # "over #NUM"
        "atmost [0-9]+",  # "over #NUM"
        "shorter than [0-9]+",  # "over #NUM"
        "longer than [0-9]+",  # "over #NUM"
        "fewer than [0-9]+",  # "over #NUM"
        "greater than [0-9]+",  # "over #NUM"
        "higher [0-9]+",  # "over #NUM"
        "lower [0-9]+",  # "over #NUM"
        "higher than [0-9]+",  # "over #NUM"
        "lower than [0-9]+",  # "over #NUM"
        "less than [0-9]+",  # "over #NUM"
        "more than [0-9]+",  # "over #NUM"
        "[0-9]+\s\w+\sor longer",   # 10 yards or longer
        "[0-9]+\s\w+\sor more",   # 10 yards or more
        "[0-9]+\s\w+\sor shorter",  # 10 yards or shorter
        "[0-9]+\s\w+\sor less",   # 10 yards or less
        "[0-9]+\s\w+\sor fewer",   # 10 yards or fewer
    ]
re_filternum_patterns = [re.compile(p) for p in patterns_for_filternum]

def is_potential_filter_num(question: str, program_node: Node, program_lisp: str) -> bool:
    match = False
    for re_pattern in re_filternum_patterns:
        if re_pattern.search(question) is not None:
            match = True

    if not any([x in question for x in ["field", "goal", "touchdown", "yard"]]):
        # Remove non-football questions
        match = False

    return match


re_num_pattern = re.compile("\s[0-9]+\s")
re_year_pattern = re.compile("\s[0-9]{4}\s")
def is_num_in_question(question: str, program_node: Node, program_lisp: str):
    match = False
    if re_num_pattern.search(question) is not None and re_year_pattern.search(question) is None:
        # Number in question but not year (4-digit number)
        match = True
    return match


def is_minmax_no_longshort(question: str, program_node: Node, program_lisp: str) -> bool:
    match = False
    if any([x in program_lisp for x in ["select_min_num", "select_max_num"]]):
        if not any([x in question for x in ["longest", "shortest"]]):
            match = True
    return match


def is_longshort_no_minmax(question: str, program_node: Node, program_lisp: str) -> bool:
    match = False
    if any([x in question for x in ["longest", "shortest"]]):
        if not any([x in program_lisp for x in ["select_min_num", "select_max_num"]]):
            match = True
    return match


def is_selectnum_select_question(question: str, program_node: Node, program_lisp: str) -> bool:
    if program_lisp == "(select_num select_passage)":
        return True


def is_passagenum_diff(question: str, program_node: Node, program_lisp: str) -> bool:
    match = False
    if program_lisp == "(passagenumber_difference (select_num select_passage) (select_num select_passage))":
        match = True
    return match


def is_passagenum_diff_not_howmanymore(question: str, program_node: Node, program_lisp: str) -> bool:
    match = False
    if program_lisp == "(passagenumber_difference (select_num select_passage) (select_num select_passage))":
        if "how many more" not in question.lower():
            match = True
    return match



def is_selectans_select_question(question: str, program_node: Node, program_lisp: str):
    match = False
    if program_node.predicate == "select_passagespan_answer" and \
            len(program_node.children) == 1 and program_node.children[0].predicate == "select_passage":
        match = True
    return match


def get_filtered_dataset(dataset: Dict, filtering_function: Callable) -> Tuple[Dict, List[Dict]]:
    """ Filter dataset to remove "select_passagespan_answer(select_passage)" questions.
    """
    filtered_data = {}
    total_qa = 0
    num_filtered_qa = 0

    for passage_id, passage_info in dataset.items():
        filtered_qas = []
        for qa in passage_info[constants.qa_pairs]:
            question = qa[constants.question]
            total_qa += 1
            if constants.program_supervision in qa:
                program_node = node_from_dict(qa[constants.program_supervision])
                program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
                kwargs = {"question": question, "program_node": program_node, "program_lisp": program_lisp}
                select_question = filtering_function(**kwargs)
                if select_question:
                    filtered_qas.append(qa)
                    num_filtered_qa += 1

                    # if filtering_function == is_passagenum_diff:
                    #     print(f"{question}  {program_node.get_nested_expression_with_strings()}")

        if filtered_qas:
            pinfo = copy.deepcopy(passage_info)
            pinfo[constants.qa_pairs] = filtered_qas
            filtered_data[passage_id] = pinfo

    json_dicts: List[Dict] = get_json_dicts(filtered_data)
    random.shuffle(json_dicts)

    print(f"Number of filtered questions: {num_filtered_qa}")
    return filtered_data, json_dicts


def year_diffs(passage_date_objs):
    year_differences = []
    for (date1, date2) in itertools.product(passage_date_objs, repeat=2):
        year_diff = date1.year_diff(date2)
        if year_diff >= 0:
            if year_diff not in year_differences:
                year_differences.append(year_diff)

    return sorted(year_differences)


def get_json_dicts(drop_dataset):
    output_json_dicts = []

    for passage_id, passage_info in drop_dataset.items():
        passage = passage_info[constants.passage]
        passage_number_values = passage_info[constants.passage_num_normalized_values]
        passage_date_values = passage_info[constants.passage_date_normalized_values]
        passage_date_objs = [Date(day=d, month=m, year=y) for (d, m, y) in passage_date_values]
        year_differences = year_diffs(passage_date_objs)

        for qa in passage_info[constants.qa_pairs]:
            query_id = qa[constants.query_id]
            question = qa[constants.question]
            answer_annotation = qa[constants.answer]
            answer_type, answers = convert_answer(answer_annotation)
            program_supervision = qa.get(constants.program_supervision, None)
            answer_passage_spans = qa[constants.answer_passage_spans]
            if program_supervision:
                program_node = node_from_dict(program_supervision)
                nested_expr = program_node.get_nested_expression_with_strings()
                nested_tuple = convert_nestedexpr_to_tuple(nested_expr)
                lisp = nested_expression_to_lisp(program_node.get_nested_expression())
                lisp = lisp.upper()   # easier to visualize in prodigy_annotate
                prodigy_lisp = program_node.get_prodigy_lisp()
            else:
                nested_expr = []
                nested_tuple = ()
                lisp = ""
                prodigy_lisp = ""

            output_dict = {
                "question": question,
                "passage": passage,
                "query_id": query_id,
                "nested_expr": nested_expr,
                "nested_tuple": nested_tuple,
                "lisp": lisp,
                "prodigy_lisp": prodigy_lisp,
                "answer_annotation": answer_annotation,
                "answer_list": answers,
                "answer_passage_spans": answer_passage_spans,
                "passage_number_values": passage_number_values,
                "passage_date_values": passage_date_values,
                "year_differences": year_differences,
            }
            output_json_dicts.append(output_dict)

    return output_json_dicts



def main(args):
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)


    diagnostics = {
        "minmax_no_longshort": 0,
        "longshort_no_minmax": 1,
        "filter_noquarterhalf": 2,
        "filternum_potential": 3,
        "project_not_how": 4,
        "project_how": 5,
        "passage_numdiff": 6,
        "selectnum_select": 7,
        "num_in_ques": 8,
        "passage_numdiff_not_howmanymore": 9
    }

    diagnosticnum_to_function = {
        0: is_minmax_no_longshort,
        1: is_longshort_no_minmax,
        2: is_filter_noquarterhalf,
        3: is_potential_filter_num,
        4: is_project_and_not_how,
        5: is_project_and_how,
        6: is_passagenum_diff,
        7: is_selectnum_select_question,
        8: is_num_in_question,
        9: is_passagenum_diff_not_howmanymore,
    }

    train_json = "drop_dataset_train.json"
    input_json = os.path.join(args.input_dir, train_json)
    input_dataset = read_drop_dataset(input_json)
    total_qa = 0
    for _, pinfo in input_dataset.items():
        total_qa += len(pinfo[constants.qa_pairs])
    print(f"Number of input passages: {len(input_dataset)}\nNumber of input questions: {total_qa}")

    for diagnostic, diagnosticnum in diagnostics.items():
        print("\nDiagnostic: {}".format(diagnostic))
        if diagnosticnum not in diagnosticnum_to_function:
            print("No filtering function found! :(")
            continue
        filtering_function = diagnosticnum_to_function[diagnosticnum]
        filtered_data, json_dicts = get_filtered_dataset(dataset=input_dataset, filtering_function=filtering_function)

        output_dir = os.path.join(args.output_root, diagnostic)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_json = os.path.join(output_dir, train_json)
        print(f"Output json: {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(filtered_data, outf, indent=4)

        output_jsonl = os.path.join(output_dir, "train.jsonl")
        print(f"Output JsonL: {output_jsonl}")
        write_jsonl(output_jsonl, json_dicts)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_root")
    args = parser.parse_args()

    main(args)

