from typing import List, Tuple, Dict, Union, Set, Set
import os
import json
import argparse
from collections import defaultdict
import itertools
from datasets.drop import constants
from allennlp.data.tokenizers import SpacyTokenizer

from semqa.utils.qdmr_utils import Node, nested_expression_to_lisp, node_from_dict, lisp_to_nested_expression, \
    read_drop_dataset, convert_answer

spacy_tokenizer = SpacyTokenizer()

def tokenize(text: str) -> List[str]:
    tokens = spacy_tokenizer.tokenize(text)
    return [t.text for t in tokens]


def replace_elements_in_list(input, orig, replacement):
    return [replacement if x == orig else x for x in input]


def node_is_select_passage(node: Node):
    return node.predicate == "select_passage"


def node_is_select_minmax_select_passage(node: Node):
    satisfies = False
    if node.predicate in ['select_min_num', 'select_max_num']:
        if node.children[0].predicate == 'select_passage':
            satisfies = True
    return satisfies

def compute_number_support(
        numbers: List[Union[int, float]],
        implicit_numbers: List[Union[int, float]] = None,
        max_number_of_numbers_to_consider: int = 2,
) -> Tuple[List[Union[int, float]], List[Union[int, float]], Dict, Dict, Set, Set]:
    """Compute the number support based on combinations of input numbers.
    This function considers all possible addition/subtraction between all pairs of numbers (even self). This forms
    the support of the possible answers. The output is a sorted list of number support.

    Args:
        numbers: input numbers -- usually passage numbers
        implicit_numbers: Extra numbers not part of the passage, but added in language. E.g. 100, 0
        max_number_of_numbers_to_consider: number of numbers to consider to combine
    Returns:
        composed_numbers: List of output composed numbers (also includes implicit numbers)
        compnumber2addcombinations: Dict[composed_number, Set(Tuple[passage_number, passage_number])]
        compnumber2subcombinations: Dict[composed_number, Set(Tuple[passage_number, passage_number])]
            Map from number to set of number combinations that can create it using the addition/sub operator.
            For example, {2: set((1,1), (0,2))} is a valid entry for addcombinations
    """
    if max_number_of_numbers_to_consider > 2:
        raise NotImplementedError

    passagenums_w_implicitnums = [x for x in numbers]
    # Adding implicit numbers here after checking if 0 is a part of original numbers so that we don't add tons of
    #  combinations of the kind x = x + 0 / x - 0
    zero_in_passage = True if 0 in numbers else False
    # Adding implicit-numbers to the input-numbers list since they can take part in composition with input-numbers.
    if implicit_numbers:
        passagenums_w_implicitnums.extend(implicit_numbers)

    composed_num_set = set()
    # Map from composed-number to list of number-combination that lead to this number from the add/sub operation
    compnumber2subcombinations = defaultdict(set)
    compnumber2addcombinations = defaultdict(set)
    nums_from_addition = set()
    nums_from_subtraction = set()
    signs = [-1, 1]
    # all_sign_combinations = list(itertools.product(signs, repeat=2))
    # Since our modules will only perform num1-num2 / num1+num2. Computation like -num1+num2 would not be done
    all_sign_combinations = [(1.0, -1.0), (1.0, 1.0)]
    for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
        # for number_combination in itertools.combinations(numbers, r=number_of_numbers_to_consider):
        for indexed_number_combination in itertools.product(
                enumerate(passagenums_w_implicitnums), repeat=number_of_numbers_to_consider
        ):
            ((idx1, num1), (idx2, num2)) = indexed_number_combination
            number_combination = (num1, num2)
            # if idx1 == idx2: continue     # Commented: 0 in support. Un-commented: 0 not in support
            # print(indexed_number_combination)
            for sign_combination in all_sign_combinations:
                value = sum([sign * num for (sign, num) in zip(sign_combination, number_combination)])
                if value >= 0:
                    # If 0 was originally in numbers then allow its combinations, o/w don't to avoid the
                    # combinations from getting bloated with x = x+0, 0+x, x-0
                    if (0 in number_combination and zero_in_passage) or (0 not in number_combination):
                        composed_num_set.add(value)
                        if sign_combination == (1, 1):
                            compnumber2addcombinations[value].add(number_combination)
                            nums_from_addition.add(value)
                        else:  # sign_combination == [1, -1]:
                            compnumber2subcombinations[value].add(number_combination)
                            nums_from_subtraction.add(value)

    composed_numbers = sorted(list(composed_num_set))

    return (composed_numbers, passagenums_w_implicitnums, compnumber2addcombinations, compnumber2subcombinations,
            nums_from_addition, nums_from_subtraction)


# def _get_numbers_for_num_select_node(select_node: Node,
#                                      passage_tokens, passage_num_mens, passage_num_idxs, passage_num_values):
#     # This node is a select_num(select_passsage) node
#     assert select_node.predicate == "select_passage"
#     select_string_arg = select_node.string_arg
#     arg_tokens = tokenize(select_string_arg)
#     relevant_number_entidxs, number_values = get_number_distribution_supervision(
#         question_tokens=arg_tokens,
#         passage_tokens=passage_tokens,
#         passage_num_mens=passage_num_mens,
#         passage_num_entidxs=passage_num_idxs,
#         passage_num_vals=passage_num_values)
#     return relevant_number_entidxs, number_values



def numdiff_aux_supervision(dataset: Dict):
    """ Aux supervision for how many yards was style questions. """

    diff_lisp = "(passagenumber_difference (select_num select_passage) (select_num select_passage))"

    total_ques = 0
    relevant_ques = 0

    numexamaples_w_nums_annotated = 0
    prog_type_dict = {}

    import pdb

    for passage_id, passage_info in dataset.items():
        passage_tokens: List[str] = passage_info[constants.passage_tokens]
        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_idxs = passage_info[constants.passage_num_entidx]
        passage_num_values = passage_info[constants.passage_num_normalized_values]

        for qa in passage_info[constants.qa_pairs]:
            total_ques += 1

            if constants.program_supervision not in qa or not qa[constants.program_supervision]:
                continue

            answer_annotation = qa[constants.answer]
            answer_type, answer_texts = convert_answer(answer_annotation)
            if answer_type != "number":
                continue

            program_node: Node = node_from_dict(qa[constants.program_supervision])
            program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
            if program_lisp != diff_lisp:
                continue

            try:
                answer_number = float(answer_texts[0])
            except:
                continue

            relevant_ques += 1

            _, _, _, compnum2subcombs, _, _ = compute_number_support(numbers=passage_num_values)

            if answer_number in compnum2subcombs and len(compnum2subcombs[answer_number]) == 1:
                num1 = list(compnum2subcombs[answer_number])[0][0]
                num2 = list(compnum2subcombs[answer_number])[0][1]
                num1_idx, num2_idx = passage_num_values.index(num1), passage_num_values.index(num2)
                numexamaples_w_nums_annotated += 1
                select_num_node_1 = program_node.children[0].children[0]
                select_num_node_2 = program_node.children[1].children[0]
                select_num_node_1.supervision["num_entidxs"] = [num1_idx]
                select_num_node_2.supervision["num_entidxs"] = [num2_idx]
                qa[constants.execution_supervised] = True

                qa[constants.program_supervision] = program_node.to_dict()

    print(f"Total num questions:{total_ques}  hmyw questions :{relevant_ques}")
    print(f"Num of QA with annotated numbers: {numexamaples_w_nums_annotated}")

    return dataset

if __name__ == "__main__":
    print(f"Auxiliary number grounding supervision for passage-num-diff questions\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    args = parser.parse_args()

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    input_dir = args.input_dir
    train_json_path = os.path.join(input_dir, train_json)
    dev_json_path = os.path.join(input_dir, dev_json)

    train_dataset = read_drop_dataset(train_json_path)
    dev_dataset = read_drop_dataset(dev_json_path)

    new_train_dataset = numdiff_aux_supervision(train_dataset)

    new_dev_dataset = numdiff_aux_supervision(dev_dataset)

    with open(train_json_path, "w") as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(dev_json_path, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written datasets w/ passage-num-diff aux-supervision")
