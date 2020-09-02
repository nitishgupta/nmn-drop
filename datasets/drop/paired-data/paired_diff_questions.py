from typing import List, Tuple, Dict, Union, Callable, Set
import os
import json
import numpy as np
import argparse
import itertools
from collections import defaultdict

from allennlp.predictors import Predictor
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data import Token

from utils import util, spacyutils
from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    read_json_dataset, read_jsonl, lisp_to_nested_expression, nested_expression_to_tree
from semqa.predictors.qgen_predictor import QuestionGenerationPredictor
from semqa.models.qgen.conditional_qgen_model import ConditionalQuestionGenerationModel
from semqa.data.dataset_readers.qgen.squad_qgen_reader import SquadConditionalQuestionGenerationReader
from semqa.domain_languages.drop_language import Date

nlp_spacy = spacyutils.getSpacyNLP()
spacy_tokenizer = SpacyTokenizer()

Entity = Tuple[int, int, str]
CharOffsets = Tuple[int, int]


def _split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += _split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = spacy_tokenizer.tokenize(text)
    tokens = split_tokens_by_hyphen(tokens)
    return tokens


def get_question_generation_predictor(model_tar_gz) -> QuestionGenerationPredictor:
    print("Loading QGen model")
    predictor = Predictor.from_path(archive_path=model_tar_gz, cuda_device=0, predictor_name="question_generation")
    return predictor


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


def get_year_difference_candidates(passage_date_objs: List[Date]) -> Tuple[List[int], np.array]:
    """ List of integers indicating all-possible year differences between the passage-dates
        If year difference is not defined (year = -1) or negative, we don't consider such date-combinations

        Returns the following:

        Returns:
        ---------
        year_differences:
            List[int] These are the possible year differences.
        year_difference_mat: Binary np.array of shape (D, D, y_d)
            Entry (i, j, k) == 1 denotes that D[i] - D[j] == year_differences[k]
    """
    num_date_objs = len(passage_date_objs)
    # Adding zero-first since it'll definitely be added and makes sanity-checking easy
    year_differences: List[int] = [0]

    yeardiff2combs = {0: []}

    # If any year is -1, we consider the year difference to be 0
    # If the year difference is negative, we consider the difference to be 0
    for (date1, date2) in itertools.product(passage_date_objs, repeat=2):
        year_diff = date1.year_diff(date2)
        if year_diff >= 0:
            if year_diff not in year_differences:
                year_differences.append(year_diff)
                yeardiff2combs[year_diff] = []
            yeardiff2combs[year_diff].append((date1, date2))

    return year_differences, yeardiff2combs


def make_qa_pair_dict(qid: str, question: str, answer_text: str, answer_type):
    """Structure of DROP data:

    {
        "para_id": {
            "passage": passage-text,
            "qa_pairs": [
                {
                    "question": ...,
                    "answer": {"number": "", "date": {"day":"", "month": "", "year": ""}, "spans":[]},
                    "query_id": qid,
                    "highlights": [],
                    "question_type": [],
                    "validated_answers": List["answer"-dict],
                    "expert_answers": [],
                    "question_tokens": [token, ....],
                    "question_charidxs": [ .... ],
                    "question_DATE_mens": [],
                    "question_DATE_men2entidx": [],
                    "question_DATE_normalized_values": [],
                    "question_NUM_mens": [],
                    "question_NUM_men2entidx": [],
                    "question_NUM_normalized_values": [],
                    "answer_passage_spans": [],
                    "answer_question_spans": [],
                    "program_supervision": node_to_dict,
                }
            ],
            "passage_tokens": [token, ...],
            "passage_charidxs": [charidx, ...],
            "passage_sent_idxs": [],
            "passage_DATE_mens": [],
            "passage_DATE_men2entidx": [],
            "passage_DATE_normalized_values": [],
            "passage_NUM_mens": [],
            "passage_NUM_men2entidx": [],
            "passage_NUM_normalized_values": []
        }
    }
    """
    q_spacy_tokens: List[Token] = tokenize(question)
    q_spacy_tokens_texts: List[str] = [t.text for t in q_spacy_tokens]
    ques_token_charidxs: List[int] = [token.idx for token in q_spacy_tokens]

    answer_dict = {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": []}
    if answer_type == "spans":
        answer_dict["spans"].append(answer_text)
    elif answer_type == "number":
        answer_dict["number"] = answer_text
    else:
        raise NotImplementedError

    validated_answers = []
    # if len(answer_texts) > 1:
    #     for i in range(1, len(answer_texts)):
    #         val_answer_dict = {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [answer_texts[i]]}
    #         validated_answers.append(val_answer_dict)

    qa_pair_dict = {
        "question": question,
        "query_id": qid,
        "answer": answer_dict,
        "highlights": [],
        "question_type": [],
        "validated_answers": validated_answers,
        "expert_answers": [],
        "question_tokens": q_spacy_tokens_texts,
        "question_charidxs": ques_token_charidxs,
        "question_DATE_mens": [],
        "question_DATE_men2entidx": [],
        "question_DATE_normalized_values": [],
        "question_NUM_mens": [],
        "question_NUM_men2entidx": [],
        "question_NUM_normalized_values": [],
        "answer_passage_spans": [],     # this should be handled by the reader
        "answer_question_spans": [],    # this should be handled by the reader
    }

    return qa_pair_dict


def generate_question(qgen_predictor, passage, answer_text, answer_start_charoffsets):
    qgen_output = qgen_predictor.predict(passage=passage,
                                         answer_text=answer_text,
                                         answer_start_charoffsets=answer_start_charoffsets)
    question = qgen_output['predicted_question']
    return question


def get_paired_questions_numdiff(qa_dict, passage_info, qgen_predictor):
    numdiff_lisp = "(passagenumber_difference (select_num select_passage) (select_num select_passage))"
    node = node_from_dict(qa_dict[constants.program_supervision])
    lisp = nested_expression_to_lisp(node.get_nested_expression())
    if lisp != numdiff_lisp:
        return None

    answer_dict = qa_dict[constants.answer]
    if not answer_dict["number"]:
        return None

    answer = float(answer_dict["number"])
    passage_num_values = passage_info[constants.passage_num_normalized_values]
    _, _, _, compnum2subcombs, _, _ = compute_number_support(numbers=passage_num_values)
    token1_idx, token2_idx = None, None
    if answer in compnum2subcombs and len(compnum2subcombs[answer]) == 1:
        # If only one number-combination leads to the answer
        num1 = list(compnum2subcombs[answer])[0][0]
        num2 = list(compnum2subcombs[answer])[0][1]
        num1_idx, num2_idx = passage_num_values.index(num1), passage_num_values.index(num2)
        men1_idxs = [i for i, entidx in enumerate(passage_info[constants.passage_num_entidx]) if entidx == num1_idx]
        men2_idxs = [i for i, entidx in enumerate(passage_info[constants.passage_num_entidx]) if entidx == num2_idx]
        if len(men1_idxs) == 1 and len(men2_idxs) == 1:
            men1_idx, men2_idx = men1_idxs[0], men2_idxs[0]
            # Men == ("string", tokenidx, num_value)
            token1_idx = passage_info[constants.passage_num_mens][men1_idx][1]
            token2_idx = passage_info[constants.passage_num_mens][men2_idx][1]
            num1_value = passage_info[constants.passage_num_mens][men1_idx][2]
            num2_value = passage_info[constants.passage_num_mens][men2_idx][2]

    if token1_idx is None or token2_idx is None:
        return None

    passage = passage_info[constants.passage]
    passage_tokens = passage_info[constants.passage_tokens]
    qid = qa_dict[constants.query_id]

    # Contrastive Question -- 1
    answer1_startchar = passage_info[constants.passage_charidxs][token1_idx]
    answer1_text = passage_tokens[token1_idx]

    contrastive_question1 = generate_question(qgen_predictor=qgen_predictor,
                                              passage=passage,
                                              answer_text=answer1_text,
                                              answer_start_charoffsets=[answer1_startchar])
    # So we don't write "fifteen" in answer_dict, instead write "15"
    num1_value = int(num1_value) if int(num1_value) == num1_value else num1_value
    number_answer_str_1 = str(num1_value)
    contrastive_qa_dict_1 = make_qa_pair_dict(qid=qid + "-contrastive-1",
                                              question=contrastive_question1,
                                              answer_text=number_answer_str_1,
                                              answer_type="number")

    program_supervision_lisp = "(select_num select_passage)"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node1: Node = nested_expression_to_tree(nested_expr)

    program_supervision = program_node1.to_dict()
    contrastive_qa_dict_1["program_supervision"] = program_supervision

    # orig_program_node = node_from_dict(progsupervision)
    orig_program_lisp = numdiff_lisp

    extra_annotation_1 = {
        "orig_program_lisp": orig_program_lisp,
        "orig_question": qa_dict[constants.question],
        "origprog_postorder_node_idx": 0,   # Left-select is postorder = 0
        "sharedprog_postorder_node_idx": 0,  # select operation for num(select)
    }
    contrastive_qa_dict_1.update(extra_annotation_1)

    # Contrastive Question -- 2
    answer2_startchar = passage_info[constants.passage_charidxs][token2_idx]
    answer2_text = passage_tokens[token2_idx]
    contrastive_question2 = generate_question(qgen_predictor=qgen_predictor,
                                              passage=passage,
                                              answer_text=passage_tokens[token2_idx],
                                              answer_start_charoffsets=[answer2_startchar])
    num2_value = int(num2_value) if int(num2_value) == num2_value else num2_value
    number_answer_str_2 = str(num2_value)
    contrastive_qa_dict_2 = make_qa_pair_dict(qid=qid + "-contrastive-2",
                                              question=contrastive_question2,
                                              answer_text=number_answer_str_2,
                                              answer_type="number")

    program_supervision_lisp = "(select_num select_passage)"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node2: Node = nested_expression_to_tree(nested_expr)

    program_supervision = program_node2.to_dict()
    contrastive_qa_dict_2["program_supervision"] = program_supervision

    # orig_program_node = node_from_dict(progsupervision)
    orig_program_lisp = numdiff_lisp

    extra_annotation_2 = {
        "orig_program_lisp": orig_program_lisp,
        "orig_question": qa_dict[constants.question],
        "origprog_postorder_node_idx": 2,  # Right-select is postorder = 0
        "sharedprog_postorder_node_idx": 0,  # select operation for num(select)
    }
    contrastive_qa_dict_2.update(extra_annotation_2)

    paired_qa_dicts = [contrastive_qa_dict_1, contrastive_qa_dict_2]
    return paired_qa_dicts


def get_paired_questions_yeardiff(qa_dict, passage_info, qgen_predictor):
    yeardiff_lisp = "(year_difference_two_events select_passage select_passage)"
    node = node_from_dict(qa_dict[constants.program_supervision])
    lisp = nested_expression_to_lisp(node.get_nested_expression())
    if lisp != yeardiff_lisp:
        return None

    answer_dict = qa_dict[constants.answer]
    if not answer_dict["number"]:
        return None

    answer = float(answer_dict["number"])
    passage_date_values = passage_info[constants.passage_date_normalized_values]
    passage_date_objs = [Date(day=d, month=m, year=y) for (d, m, y) in passage_date_values]

    _, yeardiff2combs = get_year_difference_candidates(passage_date_objs)

    date1_tokenspan, date2_tokenspan = None, None
    if answer in yeardiff2combs and len(yeardiff2combs[answer]) == 1:
        date1 = yeardiff2combs[answer][0][0]
        date2 = yeardiff2combs[answer][0][1]

        date1_idx = [i for i, date in enumerate(passage_date_objs) if date == date1][0]   # there should be one match
        date2_idx = [i for i, date in enumerate(passage_date_objs) if date == date2][0]   # there should be one match

        # Get mention ids with this date-value
        men1_idxs = [i for i, entidx in enumerate(passage_info[constants.passage_date_entidx]) if entidx == date1_idx]
        men2_idxs = [i for i, entidx in enumerate(passage_info[constants.passage_date_entidx]) if entidx == date2_idx]
        date1_tokenspan, date2_tokenspan = None, None
        if len(men1_idxs) == 1 and len(men2_idxs) == 1:
            # Only proceed if single mention with that value
            men1_idx, men2_idx = men1_idxs[0], men2_idxs[0]
            # Men == ("string", (start-tokenidx, end-tokenidx _inclusive_), date_value)
            date1_tokenspan = passage_info[constants.passage_date_mens][men1_idx][1]
            date2_tokenspan = passage_info[constants.passage_date_mens][men2_idx][1]

    if date1_tokenspan is None or date2_tokenspan is None:
        return None

    passage = passage_info[constants.passage]
    passage_tokens = passage_info[constants.passage_tokens]
    passage_charidxs = passage_info[constants.passage_charidxs]
    qid = qa_dict[constants.query_id]

    # Contrastive Question -- 1
    answer1_startchar = passage_charidxs[date1_tokenspan[0]]
    answer1_endchar = passage_charidxs[date1_tokenspan[1]] + len(passage_tokens[date1_tokenspan[1]])
    answer1_text = passage[answer1_startchar:answer1_endchar]
    contrastive_question1 = generate_question(qgen_predictor=qgen_predictor,
                                              passage=passage,
                                              answer_text=answer1_text,
                                              answer_start_charoffsets=[answer1_startchar])

    contrastive_qa_dict_1 = make_qa_pair_dict(qid=qid + "-contrastive-1",
                                              question=contrastive_question1,
                                              answer_text=answer1_text,
                                              answer_type="spans")

    program_supervision_lisp = "(select_passagespan_answer (project_passage select_passage))"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node1: Node = nested_expression_to_tree(nested_expr)

    program_supervision = program_node1.to_dict()
    contrastive_qa_dict_1["program_supervision"] = program_supervision

    # orig_program_node = node_from_dict(progsupervision)
    orig_program_lisp = yeardiff_lisp

    extra_annotation_1 = {
        "orig_program_lisp": orig_program_lisp,
        "orig_question": qa_dict[constants.question],
        "origprog_postorder_node_idx": 0,  # Left-select is postorder = 0
        "sharedprog_postorder_node_idx": 0,  # select operation for num(select)
    }
    contrastive_qa_dict_1.update(extra_annotation_1)

    # Contrastive Question -- 2
    answer2_startchar = passage_charidxs[date2_tokenspan[0]]
    answer2_endchar = passage_charidxs[date2_tokenspan[1]] + len(passage_tokens[date2_tokenspan[1]])
    answer2_text = passage[answer2_startchar:answer2_endchar]
    contrastive_question2 = generate_question(qgen_predictor=qgen_predictor,
                                              passage=passage,
                                              answer_text=answer2_text,
                                              answer_start_charoffsets=[answer2_startchar])

    contrastive_qa_dict_2 = make_qa_pair_dict(qid=qid + "-contrastive-2",
                                              question=contrastive_question2,
                                              answer_text=answer2_text,
                                              answer_type="spans")

    program_supervision_lisp = "(select_passagespan_answer (project_passage select_passage))"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node2: Node = nested_expression_to_tree(nested_expr)

    program_supervision = program_node2.to_dict()
    contrastive_qa_dict_2["program_supervision"] = program_supervision

    # orig_program_node = node_from_dict(progsupervision)
    orig_program_lisp = yeardiff_lisp

    extra_annotation_2 = {
        "orig_program_lisp": orig_program_lisp,
        "orig_question": qa_dict[constants.question],
        "origprog_postorder_node_idx": 1,  # Right-select is postorder = 0
        "sharedprog_postorder_node_idx": 0,  # select operation for num(select)
    }
    contrastive_qa_dict_2.update(extra_annotation_2)

    paired_qa_dicts = [contrastive_qa_dict_1, contrastive_qa_dict_2]
    return paired_qa_dicts


def get_contrastive_questions(drop_dataset: Dict, qgen_model_targz: str) -> Dict:
    qgen_predictor: QuestionGenerationPredictor = get_question_generation_predictor(qgen_model_targz)
    total_questions = 0

    qtype2cont = defaultdict(int)

    qtype2function = {
        "numdiff": get_paired_questions_numdiff,
        "yeardiff": get_paired_questions_yeardiff,
    }

    # new_dataset = {}
    ques_w_paired_examples = 0
    num_paired_examples = 0
    for passage_id, passage_info in drop_dataset.items():
        new_qas = []

        for qa in passage_info[constants.qa_pairs]:
            if not constants.program_supervision in qa:
                continue
            total_questions += 1

            paired_qa_dicts_1 = get_paired_questions_yeardiff(qa_dict=qa, passage_info=passage_info,
                                                              qgen_predictor=qgen_predictor)
            if paired_qa_dicts_1 is not None and len(paired_qa_dicts_1) > 1:
                ques_w_paired_examples += 1
                num_paired_examples += len(paired_qa_dicts_1)
                qa[constants.shared_substructure_annotations] = paired_qa_dicts_1
                qtype2cont["yeardiff"] += 1
                # new_qas.append(qa)

            paired_qa_dicts_2 = get_paired_questions_numdiff(qa_dict=qa, passage_info=passage_info,
                                                             qgen_predictor=qgen_predictor)
            if paired_qa_dicts_2 is not None and len(paired_qa_dicts_2) > 1:
                ques_w_paired_examples += 1
                num_paired_examples += len(paired_qa_dicts_2)
                qa[constants.shared_substructure_annotations] = paired_qa_dicts_2
                qtype2cont["numdiff"] += 1
                # new_qas.append(qa)

        # if new_qas:
        #     passage_info[constants.qa_pairs] = new_qas
        #     new_dataset[passage_id] = passage_info

    print(f"Total questions: {total_questions}  Ques w/ paired-examples:{ques_w_paired_examples}")
    print(f"Num of paired questions: {num_paired_examples}")
    # print("Output_dataset size: {}".format(len(new_dataset)))
    # return drop_dataset
    return drop_dataset


def main(args):
    qgen_model_targz = "/shared/nitishg/checkpoints/squad-paired-data/BS_6/BEAM_1/MASKQ_false/S_42/model.tar.gz"
    input_json = args.input_json

    print(f"Reading dataset: {input_json}")
    input_dataset = read_drop_dataset(input_json)

    output_dataset = get_contrastive_questions(input_dataset, qgen_model_targz)
    output_json = args.output_json
    print(f"Writing paired-examples augmented drop data to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(output_dataset, outf, indent=4)

    """
    output_json = args.output_json
    print("Preparing datset with contrastive questions")
    squad_dataset_w_contrastive_questions = get_contrastive_questions(squad_train_dataset, qgen_model_targz)

    output_json = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_drop-wcontrastive_dep.json"
    print(f"Writing squad drop-formatted data to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(squad_dataset_w_contrastive_questions, outf, indent=4)
    """


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)