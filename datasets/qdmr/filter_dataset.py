import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp

from datasets.drop import constants


def is_project_module(question: str, program_node: Node, program_lisp: str):
    global pred2count
    nested_expr_w_str = program_node.get_nested_expression_with_strings()

    if "project_passage" in program_lisp:
        pred2count["project"] += 1


def is_filter_module(question: str, program_node: Node, program_lisp: str):
    global pred2count
    if "filter_passage" in program_lisp:
        pred2count["filter"] += 1
        if not any([x in question for x in ["quarter?", "half?"]]):
            pred2count["filter_not_quarter/half"] += 1


patterns_for_filternum = [
        "over [0-9]+",  # "over #NUM"
        "under [0-9]+",  # "over #NUM"
        "at least [0-9]+",  # "over #NUM"
        "atleast [0-9]+",  # "over #NUM"
        "at most [0-9]+",  # "over #NUM"
        "atmost [0-9]+",  # "over #NUM"
        "shorter than [0-9]+",  # "over #NUM"
        "longer than [0-9]+",  # "over #NUM"
        "higher [0-9]+",  # "over #NUM"
        "lower [0-9]+",  # "over #NUM"
        "higher than [0-9]+",  # "over #NUM"
        "lower than [0-9]+",  # "over #NUM"
        "less than [0-9]+",  # "over #NUM"
        "more than [0-9]+",  # "over #NUM"
        "[0-9]+\s\w+\sor longer",   # 10 yards or longer
        "[0-9]+\s\w+\sor shorter",  # 10 yards or shorter
    ]
re_filternum_patterns = [re.compile(p) for p in patterns_for_filternum]

def is_potential_filter_num(question: str, program_node: Node, program_lisp: str):
    global pred2count
    match = False
    for re_pattern in re_filternum_patterns:
        if re_pattern.search(question) is not None:
            match = True

    if match:
        pred2count["pot_filternum"] += 1


re_num_pattern = re.compile("\s[0-9]+\s")
re_year_pattern = re.compile("\s[0-9]{4}\s")
def num_in_question(question: str, program_node: Node, program_lisp: str):
    global pred2count

    match = False
    if re_num_pattern.search(question) is not None and re_year_pattern.search(question) is None:
        # Number in question but not year (4-digit number)
        match = True

    if match:
        pred2count["num_in_ques"] += 1


def is_selectans_select_question(question: str, program_node: Node, program_lisp: str, filtertype2count: Dict):
    match = False
    if program_node.predicate == "select_passagespan_answer" and \
            len(program_node.children) == 1 and program_node.children[0].predicate == "select_passage":
        match = True
        filtertype2count["select_ans"] += 1
    return match


def is_month_days_question(question: str, program_node: Node, program_lisp: str, filtertype2count: Dict):
    match = False
    if any([x in question.lower() for x in ["how many months", "how many days"]]):
        match = True
        filtertype2count["days_months"] += 1
    return match


def is_removable_program(question: str, program_node: Node, filtering_functions: List[Callable],
                         filtertype2count: Dict):
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    kwargs = {"question": question, "program_node": program_node, "program_lisp": program_lisp,
              "filtertype2count": filtertype2count}

    remove_question = False
    for filtering_function in filtering_functions:
        match = filtering_function(**kwargs)
        remove_question = remove_question or match  # Remove question if any of the conditions match
        # if filtering_function == is_month_days_question and match:
        #     print("{} {}".format(question, program_node.get_nested_expression_with_strings()))
    return remove_question


def get_filtered_dataset(dataset: Dict) -> Dict:
    """ Filter dataset to remove "select_passagespan_answer(select_passage)" questions.
    """
    filtertype2count = defaultdict(int)
    filtered_data = {}
    total_qa = 0
    num_filtered_qa = 0

    # Any question for which any of these functions returns True will be removed from the dataset
    filtering_functions = [is_month_days_question, is_selectans_select_question]

    for passage_id, passage_info in dataset.items():
        filtered_qas = []
        for qa in passage_info[constants.qa_pairs]:
            question = qa[constants.question]
            total_qa += 1
            if constants.program_supervision not in qa:
                # Keep questions that are not program supervised
                filtered_qas.append(qa)
                num_filtered_qa += 1

            else:
                program_node = node_from_dict(qa[constants.program_supervision])
                if not is_removable_program(question, program_node, filtering_functions, filtertype2count):
                    filtered_qas.append(qa)
                    num_filtered_qa += 1

        if filtered_qas:
            pinfo = copy.deepcopy(passage_info)
            pinfo[constants.qa_pairs] = filtered_qas
            filtered_data[passage_id] = pinfo

    print()
    print(f"Filtering functions: {filtering_functions}")
    print(f"Number of input passages: {len(dataset)}\nNumber of input questions: {total_qa}")
    print(f"Number of filtered passagess: {len(filtered_data)}\nNumber of filtered questions: {num_filtered_qa}")
    print(f"Filtertype 2 count: {filtertype2count}")
    print()
    return filtered_data


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    FILES_TO_FILTER = ["drop_dataset_train.json", "drop_dataset_dev.json"]

    for filename in FILES_TO_FILTER:
        print(filename)
        input_json = os.path.join(args.input_dir, filename)
        output_json = os.path.join(args.output_dir, filename)
        print(f"Input json: {input_json}")
        print(f"OutFile: {output_json}")

        filtered_data = get_filtered_dataset(dataset=read_drop_dataset(input_json))

        print(f"Writing merged data to : {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(filtered_data, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    main(args)

