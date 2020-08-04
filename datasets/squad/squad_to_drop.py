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
    read_json_dataset, read_jsonl
from utils import util, spacyutils

from datasets.drop import constants


""" Convert SQUAD data to DROP format for NMN.

Before running this script -- 
1. Run question_to_jsonl.py -- To get squad-train-v1.1_questions.jsonl 
2. Run scripts/datasets/squad/question_constituency_parse.sh -- to get squad-train-v1.1_questions_parse.jsonl
"""

spacy_tokenizer = SpacyTokenizer()
spacy_nlp = spacyutils.getSpacyNLP()
spacy_whitespacetokenizer = spacyutils.getWhiteTokenizerSpacyNLP()


def tokenize(text):
    tokens = spacy_tokenizer.tokenize(text)
    tokens = [t.text for t in tokens]
    tokens = [x for t in tokens for x in t.split("-")]
    return tokens


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
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
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


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


def get_qa_pair_dict(qid, question, ques_tokens, answer_text, ques_parse):
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
    # question = question.replace("  ", " ")
    # ques_spacydoc = spacyutils.getSpacyDoc(question, spacy_nlp)
    # q_spacy_tokens = [t for t in ques_spacydoc]
    q_spacy_tokens = spacy_tokenizer.tokenize(question)
    q_spacy_tokens_texts: List[str] = [t.text for t in q_spacy_tokens]
    ques_token_charidxs: List[int] = [token.idx for token in q_spacy_tokens]

    # Checking if already supplied question tokens are equivalent to the re-tokenization performed here.
    # If equal; use the token-charidxs from above, otherwise ...
    if q_spacy_tokens_texts != ques_tokens:
        print("Pre-tokenization and current-tokenization are not the same")
        print(f"ques:{question}  pre-token:{ques_tokens}  currrent-tokens:{q_spacy_tokens_texts}")

    answer_dict = {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [answer_text]}

    qa_pair_dict = {
        "question": question,
        "query_id": qid,
        "answer": answer_dict,
        "highlights": [],
        "question_type": [],
        "validated_answers": [],
        "expert_answers": [],
        "question_tokens": ques_tokens,
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

    # Make program-supervision from ques_parse: List[(start_idx, end_idx (_exclusive_), "tokenized_span_text", "LABEL")]
    relevant_span_labels = ["WHNP", "WHADVP", "WHPP"]

    project_start_idx, project_end_idx = None, None
    select_start_idx, select_end_idx = None, None
    for span in ques_parse:
        if any([label == span[3] for label in relevant_span_labels]) and span[0] == 0:
            project_start_idx = span[0]
            project_end_idx = span[1] - 1  # _inclusive_

    if project_start_idx is not None:
        select_start_idx = project_end_idx + 1
        select_end_idx = len(ques_tokens) - 2  # skip last token == ?
    else:
        return None
    project_start_idx: int = project_start_idx
    project_end_idx: int = project_end_idx
    select_start_idx: int = select_start_idx
    select_end_idx: int = select_end_idx

    project_node = Node(predicate="project_passage",
                        string_arg=" ".join(ques_tokens[project_start_idx:project_end_idx + 1]))
    project_ques_attention = [1 if project_start_idx <= i <= project_end_idx else 0 for i in range(len(ques_tokens))]
    project_node.supervision["question_attention_supervision"] = project_ques_attention

    select_node = Node(predicate="select_passage",
                       string_arg=" ".join(ques_tokens[select_start_idx:select_end_idx + 1]))
    select_ques_attention = [1 if select_start_idx <= i <= select_end_idx else 0 for i in range(len(ques_tokens))]
    select_node.supervision["question_attention_supervision"] = select_ques_attention

    project_node.add_child(select_node)

    program_node = Node(predicate="select_passagespan_answer")
    program_node.add_child(project_node)

    program_supervision = program_node.to_dict()
    qa_pair_dict["program_supervision"] = program_supervision

    return qa_pair_dict


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


def convert_squad_to_drop(squad_dataset, qid2qinfo):
    print("\nConverting SQuAD data to DROP format ... ")
    squad_dataset = squad_dataset["data"]

    squad_drop_format_dataset = {}

    total_para, total_ques = 0, 0
    for article in squad_dataset:
        for para_idx, paragraph_json in enumerate(article["paragraphs"]):
            para_id = f"squad_{total_para}"
            passage_text = paragraph_json["context"]
            passage_token_texts, passage_token_charidxs, passage_sent_idxs = get_passage_tokens(passage_text)
            qa_dicts = []
            for question_answer in paragraph_json["qas"]:
                question_text = question_answer["question"].strip().replace("\n", "")
                qid = question_answer.get("id", None)
                qinfo_dict = qid2qinfo[qid]
                ques_tokens = qinfo_dict["tokens"]
                parse_spans = qinfo_dict["parse_spans"]

                answers = question_answer["answers"]
                answer_text = answers[0]["text"]
                qa_dict = get_qa_pair_dict(qid=qid, question=question_text, ques_tokens=ques_tokens,
                                           answer_text=answer_text, ques_parse=parse_spans)
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


def main(args):
    squad_json = args.squad_json
    squad_ques_jsonl = args.squad_ques_jsonl
    squad_ques_parse_jsonl = args.squad_ques_parse_jsonl

    squad_questions = read_jsonl(squad_ques_jsonl)
    squad_questions_parse = read_jsonl(squad_ques_parse_jsonl)
    qid2qinfo = map_question_parse(squad_questions, squad_questions_parse=squad_questions_parse)

    squad_dataset = read_json_dataset(squad_json)
    squad_drop_format = convert_squad_to_drop(squad_dataset=squad_dataset, qid2qinfo=qid2qinfo)

    output_json = args.output_json
    print(f"Writing squad drop-formatted data to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(squad_drop_format, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_json")
    parser.add_argument("--squad_ques_jsonl")
    parser.add_argument("--squad_ques_parse_jsonl")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)

