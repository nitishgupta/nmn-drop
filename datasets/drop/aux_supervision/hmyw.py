from typing import List, Tuple, Dict, Union
import os
import json
import argparse
from collections import defaultdict
from nltk.corpus import stopwords
from datasets.drop import constants
from allennlp.data.tokenizers import SpacyTokenizer

from semqa.utils import qdmr_utils

spacy_tokenizer = SpacyTokenizer()

def tokenize(text: str) -> List[str]:
    tokens = spacy_tokenizer.tokenize(text)
    return [t.text for t in tokens]


def replace_elements_in_list(input, orig, replacement):
    return [replacement if x == orig else x for x in input]


def node_is_select_passage(node: qdmr_utils.Node):
    return node.predicate == "select_passage"


def node_is_select_minmax_select_passage(node: qdmr_utils.Node):
    satisfies = False
    if node.predicate in ['select_min_num', 'select_max_num']:
        if node.children[0].predicate == 'select_passage':
            satisfies = True
    return satisfies


def is_relevant_program(program_node: qdmr_utils.Node):
    relevant = False
    program_type = None
    if program_node.predicate == 'select_num':
        relevant = node_is_select_minmax_select_passage(program_node.children[0])
        if relevant:
            min_max = "min" if 'min' in program_node.children[0].predicate else "max"
            program_type = "num_" + min_max

    if program_node.predicate in ['passagenumber_difference', 'passagenumber_addition']:
        diff_add = 'dif' if "difference" in program_node.predicate else 'add'
        if program_node.children[0].predicate in ['select_num'] and \
                program_node.children[1].predicate in ['select_num']:
            select_num_1_child = program_node.children[0].children[0]
            select_num_2_child = program_node.children[1].children[0]
            node1_satisfies, node2_satisfies = False, False
            min_max_num1, min_max_num2 = None, None
            if node_is_select_passage(select_num_1_child):
                node1_satisfies = True
                min_max_num1 = "num"
            if node_is_select_minmax_select_passage(select_num_1_child):
                node1_satisfies = True
                min_max_num1 = "min" if 'min' in select_num_1_child.predicate else "max"

            if node_is_select_passage(select_num_2_child):
                node2_satisfies = True
                min_max_num2 = "num"
            if node_is_select_minmax_select_passage(select_num_2_child):
                node2_satisfies = True
                min_max_num2 = "min" if 'min' in select_num_2_child.predicate else "max"

            relevant = node1_satisfies and node2_satisfies
            if relevant:
                program_type = diff_add + "_" + min_max_num1 + "_" + min_max_num2
    return relevant, program_type


def get_number_distribution_supervision(
    question_tokens,
    passage_tokens,
    passage_num_mens,
    passage_num_entidxs,
    passage_num_vals,
    num_answer=0,
    WINDOW = 10):

    # # Only supervised longest / shortest questions -- cannot do the first / last kind of questions
    # if "longest" not in question_tokens and "shortest" not in question_tokens:
    #     return None, None
    # if num_answer is None:
    #     return None, None

    # These are the relevant tokens in the question. We'd like to find numbers that are surrounded by these tokens
    # attended_tokens = [token for att, token in zip(attention, question_tokens) if att > 0]

    # print(question_tokens)

    question_tokens = replace_elements_in_list(question_tokens, "TD", "touchdown")
    question_tokens = replace_elements_in_list(question_tokens, "goals", "goal")
    question_tokens = replace_elements_in_list(question_tokens, "touchdowns", "touchdown")
    question_tokens = replace_elements_in_list(question_tokens, "passes", "pass")
    question_tokens = replace_elements_in_list(question_tokens, "TDS", "touchdown")
    question_tokens = replace_elements_in_list(question_tokens, "runs", "run")

    passage_tokens = replace_elements_in_list(passage_tokens, "TD", "touchdown")
    passage_tokens = replace_elements_in_list(passage_tokens, "goals", "goal")
    passage_tokens = replace_elements_in_list(passage_tokens, "touchdowns", "touchdown")
    passage_tokens = replace_elements_in_list(passage_tokens, "passes", "pass")
    passage_tokens = replace_elements_in_list(passage_tokens, "TDS", "touchdown")
    passage_tokens = replace_elements_in_list(passage_tokens, "runs", "run")
    passage_tokens = replace_elements_in_list(passage_tokens, "yards", "yard")

    attended_tokens = set(question_tokens)

    # Remove irrelevant tokens from attended-tokens
    irrelevant_tokens = ["'", "'s", "of", "the", "game", "games", "in", "yards", "percentage", "percentages"]
    for t in irrelevant_tokens:
        if t in attended_tokens:
            attended_tokens.remove(t)

    # Num of passage number tokens
    number_token_idxs = [x for (_, x, _) in passage_num_mens]

    relevant_number_tokenidxs = []
    relevant_number_entidxs = []
    relevant_number_values = []

    for menidx, number_token_idx in enumerate(number_token_idxs):
        # try:
        #     if passage_tokens[number_token_idx + 1] != "-" or passage_tokens[number_token_idx + 2] != "yard":
        #         continue
        # except:
        #     continue
        starting_tokenidx = max(0, number_token_idx - WINDOW)  # Inclusive
        ending_tokenidx = min(len(passage_tokens), number_token_idx + WINDOW + 1)  # Exclusive
        surrounding_passage_tokens = set(passage_tokens[starting_tokenidx:ending_tokenidx])
        intersection_tokens = surrounding_passage_tokens.intersection(attended_tokens)
        if len(intersection_tokens) > 0.7 * len(attended_tokens):
        # if intersection_tokens == attended_tokens:
            relevant_number_tokenidxs.append(number_token_idx)
            relevant_number_entidxs.append(passage_num_entidxs[menidx])
            relevant_number_values.append(passage_num_vals[passage_num_entidxs[menidx]])

    number_values = None
    if relevant_number_entidxs:
        number_values = set()
        for entidx in relevant_number_entidxs:
            number_values.add(passage_num_vals[entidx])
        number_values = list(number_values)

        # if num_answer not in number_values[0]:  # It's now a list
        #     number_grounding = None
        #     number_values = None
    # print(attended_tokens)
    # print(" ".join(passage_tokens))
    # print(number_values)

    return relevant_number_entidxs, number_values



def _get_numbers_for_num_select_node(select_node: qdmr_utils.Node,
                                     passage_tokens, passage_num_mens, passage_num_idxs, passage_num_values):
    # This node is a select_num(select_passsage) node
    assert select_node.predicate == "select_passage"
    select_string_arg = select_node.string_arg
    arg_tokens = tokenize(select_string_arg)
    relevant_number_entidxs, number_values = get_number_distribution_supervision(
        question_tokens=arg_tokens,
        passage_tokens=passage_tokens,
        passage_num_mens=passage_num_mens,
        passage_num_entidxs=passage_num_idxs,
        passage_num_vals=passage_num_values)
    return relevant_number_entidxs, number_values



def hmyw_aux_supervision(dataset: Dict, THRESHOLD: int = 10):
    """ Aux supervision for how many yards was style questions. """

    total_ques = 0
    relevant_ques = 0

    numexamaples_w_nums_annotated = 0
    prog_type_dict = {}

    for passage_id, passage_info in dataset.items():
        passage_tokens: List[str] = passage_info[constants.passage_tokens]
        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_idxs = passage_info[constants.passage_num_entidx]
        passage_num_values = passage_info[constants.passage_num_normalized_values]

        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            # question_tokenized_text = question_answer[constants.tokenized_question]
            question_tokens: List[str] = question_answer[constants.question_tokens]
            question_tokenized_text = " ".join(question_tokens)
            answer_annotation = question_answer[constants.answer]

            program_supervision = question_answer[constants.program_supervision]
            if program_supervision is None:
                # Only add auxiliary supervision
                continue

            program_node: qdmr_utils.Node = qdmr_utils.node_from_dict(program_supervision)
            nested_expr_tuple = qdmr_utils.convert_nestedexpr_to_tuple(program_node.get_nested_expression())

            # if any([x in \
            #         qdmr_utils.nested_expression_to_lisp(program_node.get_nested_expression()) for x \
            #         in ['select_num', 'select_min_num', 'select_max_num']]):
            #     relevant_templates.add(nested_expr_tuple)

            relevant, prog_type = is_relevant_program(program_node)
            if not relevant:
                continue

            try:
                answer_number = float(answer_annotation["number"])
            except:
                answer_number = None
                continue

            prog_type_dict[prog_type] = prog_type_dict.get(prog_type, 0) + 1
            relevant_ques += 1

            if len(prog_type.split("_")) == 2:
                # One of num_min or num_max
                select_node = program_node.children[0].children[0]
                relevant_number_entidxs, number_values = _get_numbers_for_num_select_node(
                    select_node, passage_tokens, passage_num_mens, passage_num_idxs, passage_num_values)
                min_max = prog_type.split("_")[1]
                if number_values:
                    if (min_max == "min" and answer_number == min(number_values)) or \
                            (min_max == "max" and answer_number == max(number_values)):
                        numexamaples_w_nums_annotated += 1

            if len(prog_type.split("_")) == 3:
                # One of diff/add _ num/min/max _ num/min/max
                diff_or_add, type1, type2 = prog_type.split("_")
                assert diff_or_add in ["dif", "add"] and type1 in ["min", "max", "num"] and \
                       type2 in ["min", "max", "num"]
                select_num_node_1 = program_node.children[0]
                select_num_node_2 = program_node.children[1]

                if type1 == "num":
                    # select_num_node == select_num(select_passage)
                    select_node = select_num_node_1.children[0]
                    relevant_number_entidxs1, number_values1 = _get_numbers_for_num_select_node(
                        select_node, passage_tokens, passage_num_mens, passage_num_idxs, passage_num_values)
                    if number_values1 and len(number_values1) != 1:
                        relevant_number_entidxs1, number_values1 = None, None
                else:
                    # select_num_node == select_num(select_min/max_num(select_passage))
                    select_node = select_num_node_1.children[0].children[0]
                    relevant_number_entidxs1, number_values1 = _get_numbers_for_num_select_node(
                        select_node, passage_tokens, passage_num_mens, passage_num_idxs, passage_num_values)

                if type2 == "num":
                    # select_num_node == select_num(select_passage)
                    select_node = select_num_node_2.children[0]
                    relevant_number_entidxs2, number_values2 = _get_numbers_for_num_select_node(
                        select_node, passage_tokens, passage_num_mens, passage_num_idxs, passage_num_values)
                    if number_values2 and len(number_values2) != 1:
                        relevant_number_entidxs2, number_values2 = None, None
                else:
                    # select_num_node == select_num(select_min/max_num(select_passage))
                    select_node = select_num_node_2.children[0].children[0]
                    relevant_number_entidxs2, number_values2 = _get_numbers_for_num_select_node(
                        select_node, passage_tokens, passage_num_mens, passage_num_idxs, passage_num_values)

                if number_values1 and number_values2:
                    if type1 == "num":
                        num1 = number_values1[0]
                    elif type1 == "min":
                        num1 = min(number_values1)
                    else: # type == "max"
                        num1 = max(number_values1)

                    if type2 == "num":
                        num2 = number_values2[0]
                    elif type2 == "min":
                        num2 = min(number_values2)
                    else: # type == "max"
                        num2 = max(number_values2)

                    pred_ans = (num1 + num2) if diff_or_add == "add" else (num1 - num2)
                    if answer_number == pred_ans:
                        if type1 == "num":
                            select_num_node_1.supervision["num_entidxs"] = relevant_number_entidxs1
                        else:
                            select_num_node_1.children[0].supervision["num_entidxs"] = relevant_number_entidxs1

                        if type2 == "num":
                            select_num_node_2.supervision["num_entidxs"] = relevant_number_entidxs2
                        else:
                            select_num_node_2.children[0].supervision["num_entidxs"] = relevant_number_entidxs2
                        question_answer[constants.execution_supervised] = True
                        numexamaples_w_nums_annotated += 1

            question_answer[constants.program_supervision] = program_node.to_dict()

    print(f"Total num questions:{total_ques}  hmyw questions :{relevant_ques}")
    print(f"Num of QA with annotated numbers: {numexamaples_w_nums_annotated}")
    print(prog_type_dict)

    return dataset


if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    args = parser.parse_args()

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    input_dir = args.input_dir
    train_json_path = os.path.join(input_dir, train_json)
    dev_json_path = os.path.join(input_dir, dev_json)

    train_dataset = qdmr_utils.read_drop_dataset(train_json_path)
    dev_dataset = qdmr_utils.read_drop_dataset(dev_json_path)

    new_train_dataset = hmyw_aux_supervision(train_dataset, THRESHOLD=10)

    new_dev_dataset = hmyw_aux_supervision(dev_dataset, THRESHOLD=10)

    with open(train_json_path, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(dev_json_path, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written datasets w/ num-compare aux-supervision")
