from typing import List, Tuple, Dict, Union, Callable
import os
import re
import json
import copy
import argparse
from collections import defaultdict

from allennlp.data import Token
from allennlp.data.tokenizers import SpacyTokenizer

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    read_json_dataset, read_jsonl, lisp_to_nested_expression, nested_expression_to_tree
from utils import util, spacyutils
from datasets.squad.squad_utils import tokenize, split_tokens_by_hyphen, add_project_select_program_supervision, \
    make_qa_pair_dict


""" Convert SQUAD data to DROP format for NMN.

Before running this script -- 
1. Run question_to_jsonl.py -- To get squad-train-v1.1_questions.jsonl 
2. Run scripts/datasets/squad/question_constituency_parse.sh -- to get squad-train-v1.1_questions_parse.jsonl
"""

spacy_tokenizer = SpacyTokenizer()
spacy_nlp = spacyutils.getSpacyNLP()


def _convert_orig_token_span_to_hyphen_tokenization_idxs(orig_token_idx_spans: List[Tuple[int, int]],
                                                         original_tokens: List[Token], tokens: List[Token]):
    token_charidxs = [token.idx for token in tokens]
    charidx2tokenidx = {}
    for tokenidx, (char_idx, t) in enumerate(zip(token_charidxs, tokens)):
        charidx2tokenidx[char_idx] = tokenidx                       # start of token
        charidx2tokenidx[char_idx + len(t.text) - 1] = tokenidx     # end of token

    # orig_token_idx_spans: (start-inclusive, end-exclusive)
    span_charidxs = [(original_tokens[x].idx, original_tokens[y - 1].idx + len(original_tokens[y - 1]))
                     for (x, y) in orig_token_idx_spans]

    span_token_idxs = [(charidx2tokenidx[start_char_idx], charidx2tokenidx[end_char_idx - 1] + 1) for
                           (start_char_idx, end_char_idx) in span_charidxs]
    return span_token_idxs


def map_question_parse(squad_questions: List[Dict], squad_questions_parse: List[Dict]):
    """Return a mapping from query-id to question, tokens, and constituency-spans."""
    assert len(squad_questions) == len(squad_questions_parse), print(f"Num of ques and ques-parse is not equal."
                                                                     f" {len(squad_questions)}"
                                                                     f" != {len(squad_questions_parse)}")
    print("Num of input questions: {}".format(len(squad_questions)))
    qid2qinfo = {}
    for qdict, qparse_dict in zip(squad_questions, squad_questions_parse):
        question = qdict["sentence"]
        qid = qdict["sentence_id"]
        qtokens = qparse_dict["tokens"]
        qparse = qparse_dict["spans"]

        qid2qinfo[qid] = {
            "question": question,
            "tokens": qtokens,
            "parse_spans": qparse
        }
    return qid2qinfo


def get_passage_tokens(passage_text):
    passage_spacydoc = spacyutils.getSpacyDoc(passage_text, spacy_nlp)
    original_passage_tokens = [t for t in passage_spacydoc]
    passage_tokens: List[Token] = split_tokens_by_hyphen(original_passage_tokens)
    passage_token_charidxs: List[int] = [token.idx for token in passage_tokens]
    passage_token_texts: List[str] = [t.text for t in passage_tokens]

    sentence_original_tokens_idxs = sorted([(sentence.start, sentence.end) for sentence in passage_spacydoc.sents],
                                           key=lambda x: x[0])
    passage_sent_idxs = _convert_orig_token_span_to_hyphen_tokenization_idxs(
        orig_token_idx_spans=sentence_original_tokens_idxs, original_tokens=original_passage_tokens,
        tokens=passage_tokens)

    return passage_token_texts, passage_token_charidxs, passage_sent_idxs





def make_squad_instance(passage_text, passage_token_texts, passage_token_charidxs, passage_sent_idxs,
                        qa_pair_dicts):
    """Structure of DROP data:

        {
            "para_id": {
                "passage": passage-text,
                "passage_tokens": [token, ...],
                "passage_charidxs": [charidx, ...],
                "passage_sent_idxs": [],
                "passage_DATE_mens": [],
                "passage_DATE_men2entidx": [],
                "passage_DATE_normalized_values": [],
                "passage_NUM_mens": [],
                "passage_NUM_men2entidx": [],
                "passage_NUM_normalized_values": [],
                "qa_pairs": [
                    {
                        "question": ...,
                        "answer": {"number": "", "date": {"day":"", "month": "", "year": ""}, "spans":[]},
                        "query_id": qid,
                        "highlights": [],
                        "question_type": [],
                        "validated_answers": [],
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
            }
        }
    """

    instance_dict = {
        "passage": passage_text,
        "passage_tokens": passage_token_texts,
        "passage_charidxs": passage_token_charidxs,
        "passage_sent_idxs": passage_sent_idxs,
        "passage_DATE_mens": [],
        "passage_DATE_men2entidx": [],
        "passage_DATE_normalized_values": [],
        "passage_NUM_mens": [],
        "passage_NUM_men2entidx": [],
        "passage_NUM_normalized_values": [],
        "qa_pairs": qa_pair_dicts
    }

    return instance_dict


def convert_squad_to_drop(squad_dataset):
    """Convert SQuAD dataset into DROP format.

    1. Tokenize passage, get token-charoffsets, and sentence-boundaries.
    2. For each question generate a qa_dict (drop_style)
    3. Using 1. and 2. make a passage_info dict (drop_style). para_ids are given serially since SQuAD doesn't have them
    """

    print("\nConverting SQuAD data to DROP format ... ")
    squad_dataset = squad_dataset["data"]

    squad_drop_format_dataset = {}

    total_para, total_ques = 0, 0
    for article in squad_dataset:
        title = article["title"]
        for para_idx, paragraph_json in enumerate(article["paragraphs"]):
            para_id = f"{title}_{para_idx}"
            passage_text = paragraph_json["context"]
            passage_token_texts, passage_token_charidxs, passage_sent_idxs = get_passage_tokens(passage_text)
            qa_dicts = []
            for question_answer in paragraph_json["qas"]:
                question_text = question_answer["question"].strip().replace("\n", "")
                qid = question_answer["id"] #.get("id", None)

                answers = question_answer["answers"]
                answer_texts = [answer["text"] for answer in answers]
                qa_dict = make_qa_pair_dict(qid=qid, question=question_text, answer_texts=answer_texts,
                                            spacy_tokenizer=spacy_tokenizer)
                qa_dict = add_project_select_program_supervision(qa_dict=qa_dict)   # add program supervision
                if qa_dict is not None:
                    qa_dicts.append(qa_dict)
                    total_ques += 1

            if not qa_dicts:
                continue

            instance_dict = make_squad_instance(passage_text=passage_text,
                                                passage_token_texts=passage_token_texts,
                                                passage_token_charidxs=passage_token_charidxs,
                                                passage_sent_idxs=passage_sent_idxs,
                                                qa_pair_dicts=qa_dicts)
            squad_drop_format_dataset[para_id] = instance_dict
            total_para += 1
            if total_para % 500 == 0:
                print("parsing status: paras:{} ques:{}".format(total_para, total_ques))

    print("Output dataset stats")
    print("Number of paragraph: {} Number of questions: {}".format(total_para, total_ques))

    return squad_drop_format_dataset


def get_squad_input_output_json(squad_datadir, train_or_dev: str):
    squad_input_json = f"squad-{train_or_dev}-v1.1.json"
    squad_drop_output_json = f"squad-{train_or_dev}-v1.1_drop.json"
    input_json = os.path.join(squad_datadir, squad_input_json)
    output_json = os.path.join(squad_datadir, squad_drop_output_json)
    return input_json, output_json


def main(args):
    squad_datadir = args.squad_datadir
    # squad_ques_jsonl = args.squad_ques_jsonl
    # squad_ques_parse_jsonl = args.squad_ques_parse_jsonl

    # squad_questions = read_jsonl(squad_ques_jsonl)
    # squad_questions_parse = read_jsonl(squad_ques_parse_jsonl)
    # qid2qinfo = map_question_parse(squad_questions, squad_questions_parse=squad_questions_parse)

    for train_or_dev in ["dev", "train"]:
        input_json, output_json = get_squad_input_output_json(squad_datadir, train_or_dev)
        print("\nReading squad data from: {}".format(input_json))
        squad_dataset = read_json_dataset(input_json)
        squad_drop_format = convert_squad_to_drop(squad_dataset=squad_dataset)

        print(f"Writing squad drop-formatted data to : {output_json}")
        with open(output_json, 'w') as outf:
            json.dump(squad_drop_format, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_datadir")
    # parser.add_argument("--squad_ques_jsonl")
    # parser.add_argument("--squad_ques_parse_jsonl")
    args = parser.parse_args()

    main(args)

