import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp

from datasets.drop import constants


superlatives = ["longest", "shortest", "farthest", "nearest", "most", "least"]
degree = ["second", "third", "fourth", "fifth"]
superlative_phrases = []
for d in degree:
    for s in superlatives:
        superlative_phrases.append(f"{d} {s}")


def is_second_superlative_question(question: str, program_node: Node, program_lisp: str, filtertype2count: Dict):
    match = False
    if any([x in question for x in superlative_phrases]):
        match = True
        filtertype2count["top-k-superlative"] += 1
    return match


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
        # if filtering_function == is_second_superlative_question and match:
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
    filtering_functions = [is_month_days_question, is_selectans_select_question, is_second_superlative_question]

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

