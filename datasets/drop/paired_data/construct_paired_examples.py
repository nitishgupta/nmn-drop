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

from datasets.drop.paired_data.generate_diff_questions import get_countselect_to_select_paired_questions, \
    get_numminmax_to_select_paired_questions, get_projectminmax_to_select_paired_questions, \
    generate_paired_questions_yeardiff, get_paired_questions_numdiff, get_paired_questions_datecompare, \
    get_num_minmax_paired_questions, get_project_minmax_paired_questions, get_strarg_paired_questions_yeardiff

spacy_tokenizer = SpacyTokenizer()

Entity = Tuple[int, int, str]
CharOffsets = Tuple[int, int]

random.seed(42)


FILTER_ARG_OPTIONS = {
        "first quarter": ["second quarter", "third quarter", "fourth quarter"],
        "second quarter": ["first quarter", "third quarter", "fourth quarter"],
        "third quarter": ["first quarter", "second quarter", "fourth quarter"],
        "fourth quarter": ["first quarter", "second quarter", "third quarter"],
        "first half": ["second half"],
        "second half": ["first half"],
    }


def generate_question(qgen_predictor, passage, answer_text, answer_start_charoffsets):
    qgen_output = qgen_predictor.predict(passage=passage,
                                         answer_text=answer_text,
                                         answer_start_charoffsets=answer_start_charoffsets)
    question = qgen_output['predicted_question']
    return question


def get_program_supervision_dict(program_lisp):
    aux_nested_expr = lisp_to_nested_expression(program_lisp)
    aux_program_node = nested_expression_to_tree(aux_nested_expr)
    aux_program_supervision = aux_program_node.to_dict()
    return aux_program_supervision


min_superlatives = ["shortest", "nearest"]
max_superlatives = ["longest", "farthest"]
football_events = ["touchdown", "field", "interception", "score", "Touchdown", "TD", "touch", "rushing", "catch",
                   "scoring", "return", "touchdowns", "interceptions", "passes"]
# Phrases like "shortest touchdown", "longest TD"
superlative_football_phrases = []
for e in football_events:
    for s in max_superlatives + min_superlatives:
        superlative_football_phrases.append(f"{s} {e}")


def pluralize(aux_question):
    if "pass" in aux_question:
        aux_question = aux_question.replace("pass", "passes")
    elif "run" in aux_question:
        aux_question = aux_question.replace("run", "runs")
    elif "touchdown" in aux_question:
        aux_question = aux_question.replace("touchdown", "touchdowns")
    elif "goal" in aux_question:
        aux_question = aux_question.replace("goal", "goals")
    elif "TD" in aux_question:
        aux_question = aux_question.replace("TD", "TDs")
    return aux_question


def get_numminmaxfilter_to_paired_questions(qa_dict, passage_info, qgen_predictor):
    """ For minmax_filter_find num(max(filter(find))) original lisps.

    We add two questions:
    1. From select arg - enforce consistency at select node
    2. Remove superlative and make a filter(select) question -- consistency at filter node
    """
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    max_lisp = "(select_num (select_max_num (filter_passage select_passage)))"
    min_lisp = "(select_num (select_min_num (filter_passage select_passage)))"

    orig_program_lisp = None
    if program_lisp == max_lisp:
        prog_type = "max"
    elif program_lisp == min_lisp:
        prog_type = "min"
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

    question_tokens = qa_dict[constants.question_tokens]
    if question_tokens[0:4] != ["How", "many", "yards", "was"]:
        return None

    # Generate select question by only using the select argument / attention
    select_node = program_node.children[0].children[0].children[0]
    if select_node.string_arg is not None:
        tokens = tokenize(select_node.string_arg, spacy_tokenizer=spacy_tokenizer)
        tokens = [t.text for t in tokens]
    elif "question_attention_supervision" in select_node.supervision:
        qattn = select_node.supervision["question_attention_supervision"]
        tokens = [t for i, t in enumerate(question_tokens) if qattn[i] == 1]
    else:
        return None

    aux_tokens = ["What", "were"] + tokens + ["?"]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    aux_select_question = " ".join(aux_tokens)
    aux_select_question = pluralize(aux_select_question)
    aux_program_lisp = "(select_passagespan_answer select_passage)"
    aux_program_supervision = get_program_supervision_dict(aux_program_lisp)

    paired_qa_dict1 = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                               question=aux_select_question,
                                               answer_text="", answer_type="spans",
                                               program_supervision=aux_program_supervision,
                                               orig_program_lisp=program_lisp,
                                               orig_question=qa_dict[constants.question],
                                               origprog_postorder_node_idx=0,
                                               sharedprog_postorder_node_idx=0,
                                               spacy_tokenizer=spacy_tokenizer)

    # Generate filter-select question
    # TODO(nitish) Only supervising filter outputs to be the same. Add multi-node capability to tie selects as well!
    # Hopefully the question is of the form: "How many yards was McNabb's shortest FG in the third quarter?"
    # We would like to generate a filter(select) question -- "What were McNabb's FG in the third quarter"
    aux_tokens = ["What", "were"] + question_tokens[4:]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    aux_filterselect_question = " ".join(aux_tokens)
    aux_filterselect_question = pluralize(aux_filterselect_question)
    aux_program_lisp2 = "(select_passagespan_answer (filter_passage select_passage))"
    aux_program_supervision2 = get_program_supervision_dict(aux_program_lisp2)
    paired_qa_dict2 = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-2",
                                               question=aux_filterselect_question,
                                               answer_text="", answer_type="spans",
                                               program_supervision=aux_program_supervision2,
                                               orig_program_lisp=program_lisp,
                                               orig_question=qa_dict[constants.question],
                                               origprog_postorder_node_idx=1,
                                               sharedprog_postorder_node_idx=1,
                                               spacy_tokenizer=spacy_tokenizer)

    paired_qa_dicts = [paired_qa_dict1, paired_qa_dict2]
    return paired_qa_dicts


def get_numminmaxfilter_divergent_paired_questions(qa_dict, passage_info, qgen_predictor):
    """ For minmax_filter_find num(max(filter(find))) original lisps.

    We add one divergent question corresponding to the filter(select) sub-tree BUT change the filter arg so that the
    outputs of the original and paired filters should be different.
    """

    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    max_lisp = "(select_num (select_max_num (filter_passage select_passage)))"
    min_lisp = "(select_num (select_min_num (filter_passage select_passage)))"

    if program_lisp == min_lisp:
        prog_type = "min"
    elif program_lisp == max_lisp:
        prog_type = "max"
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

    question_tokens = qa_dict[constants.question_tokens]
    if question_tokens[0:4] != ["How", "many", "yards", "was"]:
        return None

    # Generate filter-select question
    # TODO(nitish) Only supervising filter outputs to be the same. Add multi-node capability to tie selects as well!
    # Hopefully the question is of the form: "How many yards was McNabb's shortest FG in the third quarter?"
    # We would like to generate a filter(select) question -- "What were McNabb's FG in the fourth quarter"
    aux_tokens = ["What", "were"] + question_tokens[4:]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    filterselect_question = " ".join(aux_tokens)
    filterselect_question = pluralize(filterselect_question)

    divergent_filterselect_question = None
    for x in FILTER_ARG_OPTIONS:
        if x in filterselect_question:
            options = FILTER_ARG_OPTIONS[x]
            diff_filter_arg = random.choice(options)
            divergent_filterselect_question = filterselect_question.replace(x, diff_filter_arg)

    if divergent_filterselect_question is None:
        return None

    aux_program_lisp = "(select_passagespan_answer (filter_passage select_passage))"
    aux_program_supervision = get_program_supervision_dict(aux_program_lisp)
    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-div1",
                                              question=divergent_filterselect_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,       # Both selects should be the same
                                              sharedprog_postorder_node_idx=0,
                                              origprog_postorder_divnode_idx=1,    # Both filters should be divergent
                                              sharedprog_postorder_divnode_idx=1,
                                              spacy_tokenizer=spacy_tokenizer)

    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts

def get_projectminmaxfilter_to_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    minmax_lisps = ["(select_passagespan_answer (project_passage (select_max_num (filter_passage select_passage))))",
                    "(select_passagespan_answer (project_passage (select_min_num (filter_passage select_passage))))"]

    question = qa_dict[constants.question]
    if program_lisp not in minmax_lisps:
        return None
    if not ("shortest" in question or "longest" in question):
        return None

    if any([x in question for x in ["second longest", "third longest", "second shortest", "third shortest"]]):
        return None

    question_tokens = qa_dict[constants.question_tokens]
    # Generate select question by only using the select argument / attention
    select_node = program_node.children[0].children[0].children[0].children[0]
    if select_node.string_arg is not None:
        tokens = tokenize(select_node.string_arg, spacy_tokenizer=spacy_tokenizer)
        tokens = [t.text for t in tokens]
    elif "question_attention_supervision" in select_node.supervision:
        qattn = select_node.supervision["question_attention_supervision"]
        tokens = [t for i, t in enumerate(question_tokens) if qattn[i] == 1]
    else:
        return None

    aux_tokens = ["What", "were"] + tokens + ["?"]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    aux_select_question = " ".join(aux_tokens)
    aux_select_question = pluralize(aux_select_question)
    aux_program_lisp = "(select_passagespan_answer select_passage)"
    aux_program_supervision = get_program_supervision_dict(aux_program_lisp)

    paired_qa_dict1 = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                               question=aux_select_question,
                                               answer_text="", answer_type="spans",
                                               program_supervision=aux_program_supervision,
                                               orig_program_lisp=program_lisp,
                                               orig_question=qa_dict[constants.question],
                                               origprog_postorder_node_idx=0,
                                               sharedprog_postorder_node_idx=0,
                                               spacy_tokenizer=spacy_tokenizer)

    # Generate filter-select question
    # TODO(nitish) Only supervising filter outputs to be the same. Add multi-node capability to tie selects as well!
    # We would like to generate a filter(select) question -- "What were McNabb's FG in the third quarter"
    aux_tokens = ["What", "were"] + question_tokens[2:]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    aux_filterselect_question = " ".join(aux_tokens)
    aux_filterselect_question = pluralize(aux_filterselect_question)
    aux_program_lisp2 = "(select_passagespan_answer (filter_passage select_passage))"
    aux_program_supervision2 = get_program_supervision_dict(aux_program_lisp2)
    paired_qa_dict2 = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-2",
                                               question=aux_filterselect_question,
                                               answer_text="", answer_type="spans",
                                               program_supervision=aux_program_supervision2,
                                               orig_program_lisp=program_lisp,
                                               orig_question=qa_dict[constants.question],
                                               origprog_postorder_node_idx=1,
                                               sharedprog_postorder_node_idx=1,
                                               spacy_tokenizer=spacy_tokenizer)

    paired_qa_dicts = [paired_qa_dict1, paired_qa_dict2]
    return paired_qa_dicts


def get_projectminmaxfilter_divergent_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    minmax_lisps = ["(select_passagespan_answer (project_passage (select_max_num (filter_passage select_passage))))",
                    "(select_passagespan_answer (project_passage (select_min_num (filter_passage select_passage))))"]

    question = qa_dict[constants.question]
    if program_lisp not in minmax_lisps:
        return None
    if not ("shortest" in question or "longest" in question):
        return None

    if any([x in question for x in ["second longest", "third longest", "second shortest", "third shortest"]]):
        return None

    question_tokens = qa_dict[constants.question_tokens]
    # Generate filter-select question
    # TODO(nitish) Only supervising filter outputs to be the same. Add multi-node capability to tie selects as well!
    # We would like to generate a filter(select) question -- "What were McNabb's FG in the third quarter"
    aux_tokens = ["What", "were"] + question_tokens[2:]
    aux_tokens = [t for t in aux_tokens if t not in ["longest", "shortest"]]
    filterselect_question = " ".join(aux_tokens)
    filterselect_question = pluralize(filterselect_question)

    divergent_filterselect_question = None
    for x in FILTER_ARG_OPTIONS:
        if x in filterselect_question:
            options = FILTER_ARG_OPTIONS[x]
            diff_filter_arg = random.choice(options)
            divergent_filterselect_question = filterselect_question.replace(x, diff_filter_arg)

    if divergent_filterselect_question is None:
        return None

    aux_program_lisp = "(select_passagespan_answer (filter_passage select_passage))"
    aux_program_supervision = get_program_supervision_dict(aux_program_lisp)
    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-div1",
                                              question=divergent_filterselect_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,   # select outputs should be the same
                                              sharedprog_postorder_node_idx=0,
                                              origprog_postorder_divnode_idx=1,     # filter outputs should be divergent
                                              sharedprog_postorder_divnode_idx=1,
                                              spacy_tokenizer=spacy_tokenizer)

    paired_qa_dicts = [paired_qa_dict]
    return paired_qa_dicts


def get_countfilterselect_to_paired_questions(qa_dict, passage_info, qgen_predictor):
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

    # Generate select question by only using the select argument / attention
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
    aux_select_question = " ".join(aux_tokens)
    aux_program_lisp = "(select_passagespan_answer select_passage)"
    aux_program_supervision = get_program_supervision_dict(aux_program_lisp)
    paired_qa_dict1 = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-1",
                                               question=aux_select_question,
                                               answer_text="", answer_type="spans",
                                               program_supervision=aux_program_supervision,
                                               orig_program_lisp=program_lisp,
                                               orig_question=qa_dict[constants.question],
                                               origprog_postorder_node_idx=0,
                                               sharedprog_postorder_node_idx=0,
                                               spacy_tokenizer=spacy_tokenizer)

    # Generate filter-select question by replacing "How many"
    # TODO(nitish) Only supervising filter outputs to be the same. Add multi-node capability to tie selects as well!
    aux_tokens = ["What"] + question_tokens[2:]
    aux_filterselect_question = " ".join(aux_tokens)
    aux_program_lisp2 = "(select_passagespan_answer (filter_passage select_passage))"
    aux_program_supervision2 = get_program_supervision_dict(aux_program_lisp2)
    paired_qa_dict2 = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-2",
                                               question=aux_filterselect_question,
                                               answer_text="", answer_type="spans",
                                               program_supervision=aux_program_supervision2,
                                               orig_program_lisp=program_lisp,
                                               orig_question=qa_dict[constants.question],
                                               origprog_postorder_node_idx=1,
                                               sharedprog_postorder_node_idx=1,
                                               spacy_tokenizer=spacy_tokenizer)

    paired_qa_dicts = [paired_qa_dict1, paired_qa_dict2]
    return paired_qa_dicts


def get_countfilterselect_divergent_paired_questions(qa_dict, passage_info, qgen_predictor):
    program_node = node_from_dict(qa_dict[constants.program_supervision])
    program_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
    question = qa_dict[constants.question]

    count_select_lisp = "(aggregate_count (filter_passage select_passage))"

    if program_lisp != count_select_lisp:
        return None

    question_tokens = tokenize(question, spacy_tokenizer=spacy_tokenizer)
    question_tokens = [t.text for t in question_tokens]

    if not any([x in question_tokens for x in football_events]):
        return None

    # Generate filter-select question by replacing "How many"
    # TODO(nitish) Only supervising filter outputs to be the same. Add multi-node capability to tie selects as well!
    aux_program_lisp = "(select_passagespan_answer (filter_passage select_passage))"
    aux_program_supervision = get_program_supervision_dict(aux_program_lisp)
    aux_tokens = ["What"] + question_tokens[2:]
    filterselect_question = " ".join(aux_tokens)

    divergent_filterselect_question = None
    for x in FILTER_ARG_OPTIONS:
        if x in filterselect_question:
            options = FILTER_ARG_OPTIONS[x]
            diff_filter_arg = random.choice(options)
            divergent_filterselect_question = filterselect_question.replace(x, diff_filter_arg)

    if divergent_filterselect_question is None:
        return None

    paired_qa_dict = make_paired_qa_pair_dict(qid=qa_dict[constants.query_id] + "-contrastive-div1",
                                              question=divergent_filterselect_question,
                                              answer_text="", answer_type="spans",
                                              program_supervision=aux_program_supervision,
                                              orig_program_lisp=program_lisp,
                                              orig_question=qa_dict[constants.question],
                                              origprog_postorder_node_idx=0,    # Outputs of selects should be the same
                                              sharedprog_postorder_node_idx=0,
                                              origprog_postorder_divnode_idx=1,  # Output of filters should be divergent
                                              sharedprog_postorder_divnode_idx=1,
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
        # # "projectselect_to_count": get_projectselect_to_count_paired_questions,
        # MM
        # "num_minmax": get_num_minmax_paired_questions,
        # "project_minmax": get_project_minmax_paired_questions,
        # ND
        # "numdiff": get_paired_questions_numdiff,
        # DCYD
        # "yeardiff": get_strarg_paired_questions_yeardiff,
        # "datecompare": get_paired_questions_datecompare,
        # FGS -- to select
        "numminmaxfilter_to_select": get_numminmaxfilter_to_paired_questions,
        "numminmax_to_select": get_numminmax_to_select_paired_questions,
        "projectminmaxfilter_to_select": get_projectminmaxfilter_to_paired_questions,
        "projectminmax_to_select": get_projectminmax_to_select_paired_questions,
        # "countfilterselect_to_select": get_countfilterselect_to_paired_questions,
        # "countselect_to_select": get_countselect_to_select_paired_questions,
        # DV - Divergent
        # "numminmaxfilter_divergent": get_numminmaxfilter_divergent_paired_questions,
        # "projectminmaxfilter_divergent": get_projectminmaxfilter_divergent_paired_questions,
        # "countfilterselect_divergent": get_countfilterselect_divergent_paired_questions,
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