from typing import List, Tuple, Dict
import os
import json
import random
import argparse

from collections import defaultdict

from allennlp.data.tokenizers import SpacyTokenizer

from utils.util import tokenize
from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    lisp_to_nested_expression, nested_expression_to_tree, convert_answer
from semqa.domain_languages.drop_language import Date
from datasets.drop.paired_data.utils import compute_number_support, get_year_difference_candidates, make_paired_qa_pair_dict, \
    get_question_generation_predictor

# Imports unused but needed for allennlp; TODO: ask Matt how/why this works
from semqa.predictors.qgen_predictor import QuestionGenerationPredictor
from semqa.models.qgen.conditional_qgen_model import ConditionalQuestionGenerationModel
from semqa.data.dataset_readers.qgen.squad_qgen_reader import SquadConditionalQuestionGenerationReader

spacy_tokenizer = SpacyTokenizer()

Entity = Tuple[int, int, str]
CharOffsets = Tuple[int, int]

random.seed(42)

def generate_question(qgen_predictor, passage, answer_text, answer_start_charoffsets):
    qgen_output = qgen_predictor.predict(passage=passage,
                                         answer_text=answer_text,
                                         answer_start_charoffsets=answer_start_charoffsets)
    question = qgen_output['predicted_question']
    return question


# def get_paired_questions_numdiff(qa_dict, passage_info, qgen_predictor):
#     numdiff_lisp = "(passagenumber_difference (select_num select_passage) (select_num select_passage))"
#     node = node_from_dict(qa_dict[constants.program_supervision])
#     lisp = nested_expression_to_lisp(node.get_nested_expression())
#     if lisp != numdiff_lisp:
#         return None
#
#     answer_dict = qa_dict[constants.answer]
#     if not answer_dict["number"]:
#         return None
#
#     answer = float(answer_dict["number"])
#     passage_num_values = passage_info[constants.passage_num_normalized_values]
#     _, _, _, compnum2subcombs, _, _ = compute_number_support(numbers=passage_num_values)
#     token1_idx, token2_idx = None, None
#     num1_value, num2_value = None, None
#     if answer in compnum2subcombs and len(compnum2subcombs[answer]) == 1:
#         # If only one number-combination leads to the answer
#         num1 = list(compnum2subcombs[answer])[0][0]
#         num2 = list(compnum2subcombs[answer])[0][1]
#         num1_idx, num2_idx = passage_num_values.index(num1), passage_num_values.index(num2)
#         men1_idxs = [i for i, entidx in enumerate(passage_info[constants.passage_num_entidx]) if entidx == num1_idx]
#         men2_idxs = [i for i, entidx in enumerate(passage_info[constants.passage_num_entidx]) if entidx == num2_idx]
#         if len(men1_idxs) == 1 and len(men2_idxs) == 1:
#             men1_idx, men2_idx = men1_idxs[0], men2_idxs[0]
#             # Men == ("string", tokenidx, num_value)
#             token1_idx = passage_info[constants.passage_num_mens][men1_idx][1]
#             token2_idx = passage_info[constants.passage_num_mens][men2_idx][1]
#             num1_value = passage_info[constants.passage_num_mens][men1_idx][2]
#             num2_value = passage_info[constants.passage_num_mens][men2_idx][2]
#
#     if token1_idx is None or token2_idx is None:
#         return None
#
#     passage = passage_info[constants.passage]
#     passage_tokens = passage_info[constants.passage_tokens]
#     qid = qa_dict[constants.query_id]
#
#     # Contrastive Question -- 1
#     answer1_startchar = passage_info[constants.passage_charidxs][token1_idx]
#     answer1_text = passage_tokens[token1_idx]
#
#     contrastive_question1 = generate_question(qgen_predictor=qgen_predictor,
#                                               passage=passage,
#                                               answer_text=answer1_text,
#                                               answer_start_charoffsets=[answer1_startchar])
#     # So we don't write "fifteen" in answer_dict, instead write "15"
#     num1_value = int(num1_value) if int(num1_value) == num1_value else num1_value
#     number_answer_str_1 = str(num1_value)
#
#     program_supervision_lisp = "(select_num select_passage)"
#     nested_expr = lisp_to_nested_expression(program_supervision_lisp)
#     program_node1: Node = nested_expression_to_tree(nested_expr)
#     aux_program_supervision1 = program_node1.to_dict()
#
#     contrastive_qa_dict_1 = make_paired_qa_pair_dict(qid=qid + "-contrastive-1",
#                                                      question=contrastive_question1,
#                                                      answer_text=number_answer_str_1,
#                                                      answer_type="number",
#                                                      program_supervision=aux_program_supervision1,
#                                                      orig_program_lisp=numdiff_lisp,
#                                                      orig_question=qa_dict[constants.question],
#                                                      origprog_postorder_node_idx=0,     # Left-select is postorder = 0
#                                                      sharedprog_postorder_node_idx=0,   # select for num(select)
#                                                      spacy_tokenizer=spacy_tokenizer)
#
#     # Contrastive Question -- 2
#     answer2_startchar = passage_info[constants.passage_charidxs][token2_idx]
#     answer2_text = passage_tokens[token2_idx]
#     contrastive_question2 = generate_question(qgen_predictor=qgen_predictor,
#                                               passage=passage,
#                                               answer_text=passage_tokens[token2_idx],
#                                               answer_start_charoffsets=[answer2_startchar])
#     num2_value = int(num2_value) if int(num2_value) == num2_value else num2_value
#     number_answer_str_2 = str(num2_value)
#
#     program_supervision_lisp = "(select_num select_passage)"
#     nested_expr = lisp_to_nested_expression(program_supervision_lisp)
#     program_node2: Node = nested_expression_to_tree(nested_expr)
#     aux_program_supervision2 = program_node2.to_dict()
#
#     contrastive_qa_dict_2 = make_paired_qa_pair_dict(qid=qid + "-contrastive-2",
#                                                      question=contrastive_question2,
#                                                      answer_text=number_answer_str_2,
#                                                      answer_type="number",
#                                                      program_supervision=aux_program_supervision2,
#                                                      orig_program_lisp=numdiff_lisp,
#                                                      orig_question=qa_dict[constants.question],
#                                                      origprog_postorder_node_idx=2,     # Right-select is postorder = 0
#                                                      sharedprog_postorder_node_idx=0,   # select for num(select)
#                                                      spacy_tokenizer=spacy_tokenizer)
#
#     paired_qa_dicts = [contrastive_qa_dict_1, contrastive_qa_dict_2]
#     return paired_qa_dicts


def get_paired_questions_numdiff(qa_dict, passage_info, qgen_predictor):
    numdiff_lisp = "(passagenumber_difference (select_num select_passage) (select_num select_passage))"
    node = node_from_dict(qa_dict[constants.program_supervision])
    lisp = nested_expression_to_lisp(node.get_nested_expression())
    if lisp != numdiff_lisp:
        return None

    select1 = node.children[0].children[0]
    select1_arg = select1.string_arg
    select2 = node.children[1].children[0]
    select2_arg = select2.string_arg

    qid = qa_dict[constants.query_id]

    program_supervision_lisp = "(select_num select_passage)"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node: Node = nested_expression_to_tree(nested_expr)
    aux_program_supervision = program_node.to_dict()

    contrastive_question1 = "How many " + select1_arg

    contrastive_qa_dict_1 = make_paired_qa_pair_dict(qid=qid + "-contrastive-1",
                                                     question=contrastive_question1,
                                                     answer_text="",
                                                     answer_type="spans",
                                                     program_supervision=aux_program_supervision,
                                                     orig_program_lisp=numdiff_lisp,
                                                     orig_question=qa_dict[constants.question],
                                                     origprog_postorder_node_idx=0,     # Left-select is postorder = 0
                                                     sharedprog_postorder_node_idx=0,   # select for num(select)
                                                     spacy_tokenizer=spacy_tokenizer)

    # Contrastive Question -- 2
    contrastive_question2 = "How many " + select2_arg

    contrastive_qa_dict_2 = make_paired_qa_pair_dict(qid=qid + "-contrastive-2",
                                                     question=contrastive_question2,
                                                     answer_text="",
                                                     answer_type="spans",
                                                     program_supervision=aux_program_supervision,
                                                     orig_program_lisp=numdiff_lisp,
                                                     orig_question=qa_dict[constants.question],
                                                     origprog_postorder_node_idx=2,     # Right-select is postorder = 0
                                                     sharedprog_postorder_node_idx=0,   # select for num(select)
                                                     spacy_tokenizer=spacy_tokenizer)

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

    program_supervision_lisp = "(select_passagespan_answer (project_passage select_passage))"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node1: Node = nested_expression_to_tree(nested_expr)
    aux_program_supervision1 = program_node1.to_dict()

    orig_program_lisp = yeardiff_lisp

    contrastive_qa_dict_1 = make_paired_qa_pair_dict(qid=qid + "-contrastive-1",
                                                     question=contrastive_question1,
                                                     answer_text=answer1_text,
                                                     answer_type="spans",
                                                     program_supervision=aux_program_supervision1,
                                                     orig_program_lisp=orig_program_lisp,
                                                     orig_question=qa_dict[constants.question],
                                                     origprog_postorder_node_idx=0,  # Left-select is postorder = 0
                                                     sharedprog_postorder_node_idx=0,
                                                     spacy_tokenizer=spacy_tokenizer)

    # Contrastive Question -- 2
    answer2_startchar = passage_charidxs[date2_tokenspan[0]]
    answer2_endchar = passage_charidxs[date2_tokenspan[1]] + len(passage_tokens[date2_tokenspan[1]])
    answer2_text = passage[answer2_startchar:answer2_endchar]
    contrastive_question2 = generate_question(qgen_predictor=qgen_predictor,
                                              passage=passage,
                                              answer_text=answer2_text,
                                              answer_start_charoffsets=[answer2_startchar])

    program_supervision_lisp = "(select_passagespan_answer (project_passage select_passage))"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node2: Node = nested_expression_to_tree(nested_expr)
    aux_program_supervision2 = program_node2.to_dict()

    orig_program_lisp = yeardiff_lisp

    contrastive_qa_dict_2 = make_paired_qa_pair_dict(qid=qid + "-contrastive-2",
                                                     question=contrastive_question2,
                                                     answer_text=answer2_text,
                                                     answer_type="spans",
                                                     program_supervision=aux_program_supervision2,
                                                     orig_program_lisp=orig_program_lisp,
                                                     orig_question=qa_dict[constants.question],
                                                     origprog_postorder_node_idx=1,     # Right-select is postorder = 1
                                                     sharedprog_postorder_node_idx=0,   # select for num(select)
                                                     spacy_tokenizer=spacy_tokenizer)

    paired_qa_dicts = [contrastive_qa_dict_1, contrastive_qa_dict_2]
    return paired_qa_dicts


def get_paired_questions_datecompare(qa_dict, passage_info, qgen_predictor):
    node = node_from_dict(qa_dict[constants.program_supervision])
    lisp = nested_expression_to_lisp(node.get_nested_expression())
    if lisp not in ["(select_passagespan_answer (compare_date_lt select_passage select_passage))",
                    "(select_passagespan_answer (compare_date_gt select_passage select_passage))"]:
        return None

    select1_node = node.children[0].children[0]
    select2_node = node.children[0].children[1]

    # Apparently we've added string-arg to date-compare questions for ICLR (QDMR already has them)
    # This process is agnostic to whether the order of the events in the parse is reversed or not
    select1_arg = select1_node.string_arg
    select2_arg = select2_node.string_arg
    if not select1_arg or not select2_arg:
        return None

    qid = qa_dict[constants.query_id]
    # Contrastive Question -- 1
    if "when" not in select1_arg:
        contrastive_question1 = "When was " + select1_arg + "?"
    else:
        # QDMR string-arg already has "when was"
        contrastive_question1 = select1_arg + "?"

    program_supervision_lisp = "(select_passagespan_answer (project_passage select_passage))"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node: Node = nested_expression_to_tree(nested_expr)
    aux_program_supervision = program_node.to_dict()

    orig_program_lisp = lisp

    contrastive_qa_dict_1 = make_paired_qa_pair_dict(qid=qid + "-contrastive-1",
                                                     question=contrastive_question1,
                                                     answer_text="", answer_type="spans",
                                                     program_supervision=aux_program_supervision,
                                                     orig_program_lisp=orig_program_lisp,
                                                     orig_question=qa_dict[constants.question],
                                                     origprog_postorder_node_idx=0,  # Left-select is postorder = 0
                                                     sharedprog_postorder_node_idx=0,
                                                     spacy_tokenizer=spacy_tokenizer)

    # Contrastive Question -- 2
    if "when" not in select2_arg:
        contrastive_question2 = "When was " + select2_arg + "?"
    else:
        # QDMR string-arg already has "when was"
        contrastive_question2 = select2_arg + "?"

    contrastive_qa_dict_2 = make_paired_qa_pair_dict(qid=qid + "-contrastive-2",
                                                     question=contrastive_question2,
                                                     answer_text="", answer_type="spans",
                                                     program_supervision=aux_program_supervision,
                                                     orig_program_lisp=orig_program_lisp,
                                                     orig_question=qa_dict[constants.question],
                                                     origprog_postorder_node_idx=1,     # Right-select is postorder = 1
                                                     sharedprog_postorder_node_idx=0,   # select for num(select)
                                                     spacy_tokenizer=spacy_tokenizer)

    paired_qa_dicts = [contrastive_qa_dict_1, contrastive_qa_dict_2]
    return paired_qa_dicts


min_superlatives = ["shortest", "nearest"]
max_superlatives = ["longest", "farthest"]
football_events = ["touchdown", "field", "interception", "score", "Touchdown", "TD", "touch", "rushing", "catch",
                   "scoring", "return", "touchdowns", "interceptions", "passes"]
# Phrases like "shortest touchdown", "longest TD"
superlative_football_phrases = []
for e in football_events:
    for s in max_superlatives + min_superlatives:
        superlative_football_phrases.append(f"{s} {e}")


def get_num_minmax_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    max_lisp = "(select_num (select_max_num (select_passage))"
    min_lisp = "(select_num (select_min_num select_passage))"

    orig_program_lisp = None
    if program_lisp == min_lisp:
        prog_type = "min"
        orig_program_lisp = min_lisp
    elif program_lisp == max_lisp:
        prog_type = "max"
        orig_program_lisp = max_lisp
    else:
        prog_type = None

    if prog_type is None:
        return None

    question = qa_dict[constants.question]

    if prog_type is "min" and "shortest" not in question:
        return None

    if prog_type is "max" and "longest" not in question:
        return None

    if prog_type == "min":
        aux_question = question.replace("shortest", "longest")
        aux_program_lisp = max_lisp
    else:
        aux_question = question.replace("longest", "shortest")
        aux_program_lisp = min_lisp

    aux_nested_expr = lisp_to_nested_expression(aux_program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)
    aux_program_supervision = aux_program_node.to_dict()

    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                              question=aux_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=orig_program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,
                                              sharedprog_postorder_node_idx=0,
                                              spacy_tokenizer=spacy_tokenizer)
    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts


def get_numminmax_to_select_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    max_lisp = "(select_num (select_max_num (filter_passage select_passage)))"
    min_lisp = "(select_num (select_min_num (filter_passage select_passage)))"

    orig_program_lisp = None
    if program_lisp == min_lisp:
        prog_type = "min"
        orig_program_lisp = min_lisp
    elif program_lisp == max_lisp:
        prog_type = "max"
        orig_program_lisp = max_lisp
    else:
        prog_type = None

    if prog_type is None:
        return None

    question = qa_dict[constants.question]

    if prog_type is "min" and "shortest" not in question:
        return None

    if prog_type is "max" and "longest" not in question:
        return None

    if any([x in question for x in ["second longest", "third longest", "second shortest", "third shortest"]]):
        return None

    # Hopefully the question is of the form: "How many yards was McNabb's shortest FG in the third quarter?"
    # We would like to generate a (select) question -- "What were McNabb's FG in the third quarter"
    question_tokens = tokenize(question, spacy_tokenizer=spacy_tokenizer)
    question_tokens = [t.text for t in question_tokens]

    if question_tokens[0:4] != ["How", "many", "yards", "was"]:
        return None

    select_node = program_node.children[0].children[0].children[0]
    if select_node.string_arg is not None:
        tokens = tokenize(select_node.string_arg, spacy_tokenizer=spacy_tokenizer)
    elif "question_attention_supervision" in select_node.supervision:
        qattn = select_node.supervision["question_attention_supervision"]
        tokens = [t for i, t in enumerate(question_tokens) if qattn[i] == 1]
    else:
        return None

    aux_tokens = ["What", "were"] + tokens + ["?"]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    aux_question = " ".join(aux_tokens)

    if "pass" in aux_question:
        aux_question = aux_question.replace("pass", "passes")
    elif "run" in aux_question:
        aux_question = aux_question.replace("run", "runs")
    elif "touchdown" in aux_question:
        aux_question = aux_question.replace("touchdown", "touchdowns")
    elif "goal" in aux_question:
        aux_question = aux_question.replace("goal", "goals")

    print(question)
    print(aux_question)

    aux_program_lisp = "(select_passagespan_answer select_passage)"
    # aux_program_lisp = "(select_passage)"
    aux_nested_expr = lisp_to_nested_expression(aux_program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)
    aux_program_supervision = aux_program_node.to_dict()

    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-S",
                                              question=aux_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=orig_program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,
                                              sharedprog_postorder_node_idx=0,
                                              spacy_tokenizer=spacy_tokenizer)
    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts


def get_project_minmax_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    minmax_lisps = ["(select_passagespan_answer (project_passage (select_max_num select_passage)))",
                    "(select_passagespan_answer (project_passage (select_min_num select_passage)))"]

    question = qa_dict[constants.question]
    if program_lisp not in minmax_lisps:
        return None
    if not ("shortest" in question or "longest" in question):
        return None

    orig_program_lisp = program_lisp

    # Draw a min/max program and replace the "Who scored" with "How many yards was"
    max_lisp = "(select_num (select_max_num select_passage))"
    min_lisp = "(select_num (select_min_num select_passage))"

    if random.random() < 0.5:
        aux_program_lisp = min_lisp
        aux_question = question.replace("longest", "shortest")
        # aux_question_tokens = [x if x != "longest" else "shortest" for x in question_tokens] # map longest to shortest
        # superlative_index = aux_question_tokens.index("shortest")
    else:
        aux_program_lisp = max_lisp
        aux_question = question.replace("shortest", "longest")
        # aux_question_tokens = [x if x != "shortest" else "longest" for x in question_tokens]  # map shortest to longest
        # superlative_index = aux_question_tokens.index("longest")

    aux_question_tokens = tokenize(aux_question, spacy_tokenizer=spacy_tokenizer)
    aux_question_tokens = [t.text for t in aux_question_tokens]
    # replace first two "Who scored" tokens
    aux_question_tokens = ["How", "many", "yards", "was"] + aux_question_tokens[2:]
    aux_question = " ".join(aux_question_tokens)

    aux_nested_expr = lisp_to_nested_expression(aux_program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)
    aux_program_supervision = aux_program_node.to_dict()

    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                              question=aux_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,
                                              sharedprog_postorder_node_idx=0,
                                              spacy_tokenizer=spacy_tokenizer)
    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts


def get_projectminmax_to_select_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    minmax_lisps = ["(select_passagespan_answer (project_passage (select_max_num select_passage)))",
                    "(select_passagespan_answer (project_passage (select_min_num select_passage)))"]

    question = qa_dict[constants.question]
    if program_lisp not in minmax_lisps:
        return None
    if not ("shortest" in question or "longest" in question):
        return None

    if any([x in question for x in ["second longest", "third longest", "second shortest", "third shortest"]]):
        return None

    # Hopefully the question is of the form: "Who kicked the shortest FG in the third quarter?"
    # We would like to generate a (select) question -- "What were FG in the third quarter"
    question_tokens = tokenize(question, spacy_tokenizer=spacy_tokenizer)
    question_tokens = [t.text for t in question_tokens]

    aux_tokens = ["What", "were"] + question_tokens[2:]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    aux_question = " ".join(aux_tokens)

    if "pass" in aux_question:
        aux_question = aux_question.replace("pass", "passes")
    elif "run" in aux_question:
        aux_question = aux_question.replace("run", "runs")
    elif "touchdown" in aux_question:
        aux_question = aux_question.replace("touchdown", "touchdowns")
    elif "goal" in aux_question:
        aux_question = aux_question.replace("goal", "goals")

    aux_program_lisp = "(select_passagespan_answer select_passage)"
    # aux_program_lisp = "(select_passage)"
    aux_nested_expr = lisp_to_nested_expression(aux_program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)
    aux_program_supervision = aux_program_node.to_dict()

    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                              question=aux_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,
                                              sharedprog_postorder_node_idx=0,
                                              spacy_tokenizer=spacy_tokenizer)
    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts

def get_projectselect_to_count_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())

    question = qa_dict[constants.question]
    if program_lisp != "(select_passagespan_answer (project_passage select_passage))":
        return None

    answer_dict = qa_dict[constants.answer]
    answer_type, answer_texts = convert_answer(answer_dict)
    if answer_type != "spans" or len(answer_texts) == 1:
        return None

    question_tokens = tokenize(question, spacy_tokenizer)
    question_tokens = [t.text for t in question_tokens]

    if "How long were each" in question:
        aux_question = question.replace("How long were each", "How many")
        aux_program_lisp = "(aggregate_count select_passage)"
    else:
        # Replace Wh-token with How many
        aux_question_tokens = ["How", "many"] + question_tokens[1:]
        aux_question = " ".join(aux_question_tokens)
        aux_program_lisp = "(aggregate_count (project_passage select_passage))"

    aux_answer = str(len(answer_texts))

    aux_nested_expr = lisp_to_nested_expression(aux_program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)
    aux_program_supervision = aux_program_node.to_dict()

    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                              question=aux_question,
                                              answer_text=aux_answer, answer_type="number",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,
                                              sharedprog_postorder_node_idx=0,
                                              spacy_tokenizer=spacy_tokenizer)
    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts


def get_countselect_to_select_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    question = qa_dict[constants.question]

    count_select_lisp = "(aggregate_count (filter_passage select_passage))"
    # count_select_lisp = "(aggregate_count select_passage)"

    if program_lisp != count_select_lisp:
        return None

    question_tokens = tokenize(question, spacy_tokenizer=spacy_tokenizer)
    question_tokens = [t.text for t in question_tokens]

    if not any([x in question_tokens for x in football_events]):
        return None

    select_node = program_node.children[0].children[0]
    if select_node.string_arg is not None:
        tokens = tokenize(select_node.string_arg, spacy_tokenizer=spacy_tokenizer)
        tokens = [t.text for t in tokens]
    elif "question_attention_supervision" in select_node.supervision:
        qattn = select_node.supervision["question_attention_supervision"]
        tokens = [t for i, t in enumerate(question_tokens) if qattn[i] == 1]
    else:
        return None

    aux_tokens = ["What"] + tokens + ["?"]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    aux_question = " ".join(aux_tokens)

    # print(question)
    # print(aux_question)
    # print()

    # aux_tokens = ["What"] + question_tokens[2:]
    # aux_question = " ".join(aux_tokens)

    aux_program_lisp = "(select_passagespan_answer select_passage)"
    # aux_program_lisp = "(select_passage)"
    aux_nested_expr = lisp_to_nested_expression(aux_program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)
    aux_program_supervision = aux_program_node.to_dict()

    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                              question=aux_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,
                                              sharedprog_postorder_node_idx=0,
                                              spacy_tokenizer=spacy_tokenizer)
    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts


def get_contrastive_questions(drop_dataset: Dict, qgen_model_targz: str) -> Tuple[Dict, Dict]:
    # BART based question generator trained on SQuAD
    qgen_predictor = None
    # qgen_predictor: QuestionGenerationPredictor = get_question_generation_predictor(qgen_model_targz)
    total_questions = 0

    qtype2count = defaultdict(int)

    qtype2function = {
        # "numdiff": get_paired_questions_numdiff,
        # "yeardiff": get_paired_questions_yeardiff,
        # "num_minmax": get_num_minmax_paired_questions,
        # "project_minmax": get_project_minmax_paired_questions,
        # "numminmax_to_select": get_numminmax_to_select_paired_questions,
        # "projectminmax_to_select": get_projectminmax_to_select_paired_questions,
        "countselect_to_select": get_countselect_to_select_paired_questions,
        # # "projectselect_to_count": get_projectselect_to_count_paired_questions,
        # "datecompare": get_paired_questions_datecompare,
    }

    print("\nPaired examples for qtypes: {}".format(qtype2function.keys()))
    print("Total paras: {}".format(len(drop_dataset)))
    ques_w_paired_examples = 0
    num_paired_examples = 0
    paras_done = 0

    for passage_id, passage_info in drop_dataset.items():
        paras_done += 1
        for qa in passage_info[constants.qa_pairs]:
            if not constants.program_supervision in qa:
                continue
            total_questions += 1
            paired_qa_dicts = []
            for qtype, paired_question_func in qtype2function.items():
                func_qa_dicts = paired_question_func(qa_dict=qa,
                                                     passage_info=passage_info,
                                                     qgen_predictor=qgen_predictor)
                if func_qa_dicts is not None and len(func_qa_dicts) > 0:
                    qtype2count[qtype] += 1
                    paired_qa_dicts.extend(func_qa_dicts)

            if paired_qa_dicts:
                num_paired_examples += len(paired_qa_dicts)
                qa[constants.shared_substructure_annotations] = paired_qa_dicts
                ques_w_paired_examples += 1
        if paras_done % 500 == 0:
            print("paras done: {}".format(paras_done))

    print(f"Qtype2paircount: {qtype2count}")
    print(f"Total questions: {total_questions}  Ques w/ paired-examples:{ques_w_paired_examples}")
    print(f"Num of paired questions: {num_paired_examples}")

    qtype2count["total_questions"] = total_questions
    qtype2count["ques_w_paired_examples"] = ques_w_paired_examples
    qtype2count["num_paired_examples"] = num_paired_examples

    return drop_dataset, qtype2count


def main(args):
    qgen_model_targz = "/shared/nitishg/checkpoints/squad-qgen/BS_6/BEAM_1/MASKQ_false/S_42/model.tar.gz"
    input_json = args.input_json

    print("\nAugmentating data with paired-questions ... ")

    print(f"Reading dataset: {input_json}")
    input_dataset = read_drop_dataset(input_json)

    output_dataset, stats_dict = get_contrastive_questions(input_dataset, qgen_model_targz)
    output_json = args.output_json

    output_dir, output_filename = os.path.split(output_json)
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_json = os.path.join(stats_dir, output_filename)

    print(f"\nWriting paired-examples augmented drop data to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(output_dataset, outf, indent=4)

    with open(stats_json, 'w') as outf:
        json.dump(stats_dict, outf, indent=4)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)