from typing import List, Tuple, Dict
import os
import json
import copy
import random
import argparse

from collections import defaultdict

from allennlp.data.tokenizers import SpacyTokenizer

from utils.util import tokenize
from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    lisp_to_nested_expression, nested_expression_to_tree, convert_answer
from semqa.domain_languages.drop_language import Date
from datasets.drop.paired_data.utils import make_paired_qa_pair_dict, get_question_generation_predictor

# Imports unused but needed for allennlp; TODO: ask Matt how/why this works
from semqa.predictors.qgen_predictor import QuestionGenerationPredictor
from semqa.models.qgen.conditional_qgen_model import ConditionalQuestionGenerationModel
from semqa.data.dataset_readers.qgen.squad_qgen_reader import SquadConditionalQuestionGenerationReader

spacy_tokenizer = SpacyTokenizer()

random.seed(42)

def generate_question(qgen_predictor, passage, answer_text, answer_start_charoffsets):
    qgen_output = qgen_predictor.predict(passage=passage,
                                         answer_text=answer_text,
                                         answer_start_charoffsets=answer_start_charoffsets)
    question = qgen_output['predicted_question']
    return question


def get_num_questions(passage_id: str, passage_info: Dict, max_num_ques: int,
                      qgen_predictor: QuestionGenerationPredictor):
    passage = passage_info[constants.passage]
    passage_tokens = passage_info[constants.passage_tokens]
    passage_charidxs = passage_info[constants.passage_charidxs]
    # Men == ("string", tokenidx, num_value) -- number_mentions are single tokens
    passage_num_mens = copy.deepcopy(passage_info[constants.passage_num_mens])

    # Men == ("string", (start-tokenidx, end-tokenidx _inclusive_), date_value)
    date_mens = copy.deepcopy(passage_info[constants.passage_date_mens])
    date_spans = [(x, y) for (_, (x, y), _) in date_mens]

    pruned_passage_num_mens = []
    for num_men in passage_num_mens:
        tokenidx = num_men[1]
        overlap = False
        for (x, y) in date_spans:
            if x <= tokenidx <= y:
                overlap = True
        if not overlap:
            pruned_passage_num_mens.append(num_men)

    random.shuffle(pruned_passage_num_mens)

    chosen_num_mentions = pruned_passage_num_mens[:max_num_ques]

    program_supervision_lisp = "(select_num select_passage)"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node: Node = nested_expression_to_tree(nested_expr)
    aux_program_supervision = program_node.to_dict()

    num_ques_added = 0
    aux_qa_dicts = []
    for num_men in chosen_num_mentions:
        num_ques_added += 1
        # This number mention is an answer now
        token_idx = num_men[1]
        num_value = num_men[2]
        answer_startchar = passage_charidxs[token_idx]
        answer_text = passage_tokens[token_idx]
        # Using this value as answer for answer_dict -- to map "fifteen" --> "15"
        num_value = int(num_value) if int(num_value) == num_value else num_value
        answer_str = str(num_value)
        qid = passage_id + "-num-augment-" + str(num_ques_added)

        aux_question = generate_question(qgen_predictor=qgen_predictor,
                                         passage=passage,
                                         answer_text=answer_text,
                                         answer_start_charoffsets=[answer_startchar])

        aux_qa_dict = make_paired_qa_pair_dict(qid=qid,
                                               question=aux_question,
                                               answer_text=answer_str,
                                               answer_type="number",
                                               program_supervision=aux_program_supervision,
                                               orig_program_lisp="",
                                               orig_question="",
                                               origprog_postorder_node_idx=0,     # Left-select is postorder = 0
                                               sharedprog_postorder_node_idx=0,   # select for num(select)
                                               spacy_tokenizer=spacy_tokenizer)

        aux_qa_dict.pop("orig_program_lisp")
        aux_qa_dict.pop("orig_question")
        aux_qa_dict.pop("origprog_postorder_node_idx")
        aux_qa_dict.pop("sharedprog_postorder_node_idx")

        aux_qa_dicts.append(aux_qa_dict)

    return aux_qa_dicts


def get_date_questions(passage_id: str, passage_info: Dict, max_num_ques: int,
                       qgen_predictor: QuestionGenerationPredictor):
    passage = passage_info[constants.passage]
    passage_tokens = passage_info[constants.passage_tokens]
    passage_charidxs = passage_info[constants.passage_charidxs]

    # Men == ("string", (start-tokenidx, end-tokenidx _inclusive_), date_value)
    date_mens = copy.deepcopy(passage_info[constants.passage_date_mens])
    random.shuffle(date_mens)

    chosen_date_mentions = date_mens[:max_num_ques]

    program_supervision_lisp = "(select_passagespan_answer (project_passage select_passage))"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node: Node = nested_expression_to_tree(nested_expr)
    aux_program_supervision = program_node.to_dict()

    num_ques_added = 0
    aux_qa_dicts = []
    for date_men in chosen_date_mentions:
        num_ques_added += 1
        date_tokenspan = date_men[1]
        answer_startchar = passage_charidxs[date_tokenspan[0]]
        answer_endchar = passage_charidxs[date_tokenspan[1]] + len(passage_tokens[date_tokenspan[1]])
        answer_text = passage[answer_startchar:answer_endchar]
        qid = passage_id + "-date-augment-" + str(num_ques_added)

        aux_question = generate_question(qgen_predictor=qgen_predictor,
                                         passage=passage,
                                         answer_text=answer_text,
                                         answer_start_charoffsets=[answer_startchar])

        aux_qa_dict = make_paired_qa_pair_dict(qid=qid,
                                               question=aux_question,
                                               answer_text=answer_text,
                                               answer_type="spans",
                                               program_supervision=aux_program_supervision,
                                               orig_program_lisp="",
                                               orig_question="",
                                               origprog_postorder_node_idx=0,     # Left-select is postorder = 0
                                               sharedprog_postorder_node_idx=0,   # select for num(select)
                                               spacy_tokenizer=spacy_tokenizer)

        aux_qa_dict.pop("orig_program_lisp")
        aux_qa_dict.pop("orig_question")
        aux_qa_dict.pop("origprog_postorder_node_idx")
        aux_qa_dict.pop("sharedprog_postorder_node_idx")

        aux_qa_dicts.append(aux_qa_dict)

    return aux_qa_dicts


def get_contrastive_questions(drop_dataset: Dict, qgen_model_targz: str, max_num_ques: int = 3) -> Tuple[Dict, Dict]:
    # BART based question generator trained on SQuAD
    # qgen_predictor = None
    qgen_predictor: QuestionGenerationPredictor = get_question_generation_predictor(qgen_model_targz)
    total_questions = 0

    qtype2count = defaultdict(int)

    qtype2function = {
        "num": get_num_questions,
        "date": get_date_questions,
    }

    print("Paired examples for qtypes: {}".format(qtype2function.keys()))

    passages_w_questions = 0
    num_aux_examples = 0
    num_passages = len(drop_dataset)
    print("total num of passages: {}".format(num_passages))
    passages_done = 0

    new_dataset = {}
    for passage_id, passage_info in drop_dataset.items():
        passages_done += 1
        if passages_done % 100 == 0:
            print("passages_done: {}".format(passages_done))

        if "history" not in passage_id:
            continue

        aux_qa_dicts = []
        for qtype, paired_question_func in qtype2function.items():
            func_qa_dicts = paired_question_func(passage_id=passage_id,
                                                 passage_info=passage_info,
                                                 max_num_ques=max_num_ques,
                                                 qgen_predictor=qgen_predictor)
            for qa in func_qa_dicts:
                qa[constants.augmented_example] = True

            if func_qa_dicts:
                aux_qa_dicts.extend(func_qa_dicts)
                qtype2count[qtype] += len(func_qa_dicts)
                num_aux_examples += len(func_qa_dicts)

        if aux_qa_dicts:
            passages_w_questions += 1
            passage_info[constants.qa_pairs] = aux_qa_dicts
            new_dataset[passage_id] = passage_info

    print("Total passages: {}  Passages w questions: {}".format(num_passages, passages_w_questions))
    print(f"Qtype2paircount: {qtype2count}")
    print(f"Total aux questions: {num_aux_examples}")

    qtype2count["orig_num_passages"] = num_passages
    qtype2count["passages_in_augmented_data"] = passages_w_questions
    qtype2count["num_aux_examples"] = num_aux_examples

    return new_dataset, qtype2count


def main(args):
    qgen_model_targz = "/shared/nitishg/checkpoints/squad-qgen/BS_6/BEAM_1/MASKQ_false/S_42/model.tar.gz"
    input_json = args.input_json

    print("\nAugmentating data with num/date-questions ... ")

    print(f"Reading dataset: {input_json}")
    input_dataset = read_drop_dataset(input_json)

    output_dataset, stats_dict = get_contrastive_questions(input_dataset, qgen_model_targz)
    output_json = args.output_json

    output_dir, output_filename = os.path.split(output_json)
    os.makedirs(output_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_json = os.path.join(stats_dir, output_filename)

    print(f"\nWriting augmented-examples drop data to : {output_json}")
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