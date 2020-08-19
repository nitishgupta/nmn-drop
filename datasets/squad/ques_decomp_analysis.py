from typing import List, Tuple, Dict, Union, Callable
import os
import re
import json
import copy
import random
import argparse
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

from allennlp.data import Token
from allennlp.data.tokenizers import SpacyTokenizer

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    read_json_dataset, read_jsonl
from utils import util, spacyutils


random.seed(42)

""" Convert SQUAD data to DROP format for NMN.

Before running this script -- 
1. Run question_to_jsonl.py -- To get squad-train-v1.1_questions.jsonl 
2. Run scripts/datasets/squad/question_constituency_parse.sh -- to get squad-train-v1.1_questions_parse.jsonl
"""

spacy_tokenizer = SpacyTokenizer()
spacy_nlp = spacyutils.getSpacyNLP()

WHTOKENS = ["who", "where", "when", "how", "whom", "which", "what", "whose", "why"]
WH_PHRASE_LABELS = ["WHNP", "WHADVP", "WP", "WHPP", "WRB", "WDT", "WHADJP"]
dep_token_idx = "token_index"
dep_node_type = "nodeType"
dep_link = "link"
dep_children = "children"


@dataclass
class Question:
    qid: str
    question_str: str
    tokens: List[str]
    pos: List[str]
    conparse_spans: List   # List[(start, end(ex), text, LABEL)]
    depparse_tree: Dict     # {"nodeType", "link", "token_index", "children": List[Dict]}
    length: int



def map_question_parse(squad_questions: List[Dict],
                       squad_questions_conparse: List[Dict],
                       squad_questions_depparse: List[Dict]) -> Dict[str, Question]:
    """Return a mapping from query-id to Question"""
    assert len(squad_questions) == len(squad_questions_conparse), print(f"Num of ques and con-parse is not equal."
                                                                        f" {len(squad_questions)}"
                                                                        f" != {len(squad_questions_conparse)}")
    assert len(squad_questions) == len(squad_questions_depparse), print(f"Num of ques and dep-parse is not equal."
                                                                        f" {len(squad_questions)}"
                                                                        f" != {len(squad_questions_depparse)}")

    print("Num of input questions: {}".format(len(squad_questions)))
    qid2question = {}
    for qdict, conparse_dict, depparse_dict in zip(squad_questions, squad_questions_conparse, squad_questions_depparse):
        question = qdict["sentence"]
        qid = qdict["sentence_id"]
        qtokens = conparse_dict["tokens"]
        conparse_spans: List = conparse_dict["spans"]
        pos = depparse_dict["pos"]
        depparse_tree: Dict = depparse_dict["hierplane_tree"]["root"]
        assert len(qtokens) == len(pos)
        num_tokens = len(qtokens)


        qobj: Question = Question(qid=qid, question_str=question, tokens=qtokens, pos=pos, conparse_spans=conparse_spans,
                                  depparse_tree=depparse_tree, length=num_tokens)
        qid2question[qid] = qobj
    return qid2question



def get_wh_phrase(question: Question):
    return question.conparse_spans[1]


def get_second_parent_span(question: Question):
    """Get the second parent span after the first one (first one -- one that starts at 0)"""
    first_span = question.conparse_spans[1]     # 0th is the full sentence
    first_span_end = first_span[1] # _exclusive_
    # Since spans are sorted in in-order traversal, left-to-right, we can just look for the first span after this that
    # starts at first_span_end
    second_span, location = None, None
    for i in range(2, len(question.conparse_spans)):
        if question.conparse_spans[i][0] == first_span_end:
            second_span = question.conparse_spans[i]
            location = i
            break
    if second_span is None:
        raise Exception
    return second_span, i


def is_span_whtoken(span: Tuple[int, int, str, str]):
    span_text = span[2].lower()
    if any([x == span_text.lower() for x in WHTOKENS]):
        return True
    else:
        return False


def get_subspans(span: Tuple[int, int, str, str],
                 conparse_spans: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
    start, end = span[0], span[1]
    subspans = []
    for conspan in conparse_spans:
        if conspan[0] >= start and conspan[1] <= end:
            if conspan[0] == start and conspan[1] == end:
                continue
            subspans.append(conspan)
    return subspans


def get_dependencies(node: Dict, question):
    dependencies = []
    if dep_children in node:
        children: List[Dict] = node[dep_children]
        for child in children:
            token_idx = child[dep_token_idx]
            link = child[dep_link]
            pos = question.pos[token_idx]
            token = question.tokens[token_idx]
            dependencies.append((token_idx, token, pos, link))
    return dependencies


def has_nsubj(root: Dict):
    children = root.get(dep_children, None)
    if children is None:
        return False

    for c in children:
        if c[dep_link] == "nsubj" and c[dep_token_idx] != 0:
            return True


def has_dobj(root: Dict):
    children = root.get(dep_children, None)
    if children is None:
        return False

    for c in children:
        if c[dep_link] == "dobj":
            return True


def wh_span_distribution(qid2question: Dict[str, Question]):
    whspantype2count = defaultdict(int)
    num_whtoken_in_whphrase = 0
    num_whlabel = 0

    questions = list(qid2question.items())
    random.shuffle(questions)

    for qid, question in questions:
        wh_span = get_wh_phrase(question)
        start, end, text, label = wh_span
        whspantype2count[label] += 1
        if label in WH_PHRASE_LABELS:
            num_whlabel += 1

        if label == "SBAR":
            print(question.question_str)
            # print(question.conparse_spans)

    whspantype2count = util.sortedDictByValue(whspantype2count, decreasing=True)

    print(f"Num questions: {len(qid2question)}")
    print(whspantype2count)
    print(f"First span is wh-label: {num_whlabel}")
    print(f"Wh-token in wh-phrase: {num_whtoken_in_whphrase}")



def process_WHNP_span(qid2question: Dict[str, Question]):
    num_whnp = 0
    wh_phrase_is_wh_token = 0
    root_pos_dist = defaultdict(int)
    wdobj_wo_nsubj, wnsubj_wo_dobj = 0, 0
    noun_child = 0

    questions = list(qid2question.items())
    random.shuffle(questions)

    for qid, question in questions:
        # Currently this is the first span in cons-parse; could change any time. Write code accordingly
        wh_span = get_wh_phrase(question)
        start, end, text, label = wh_span
        if label != "WHNP":
            continue
        num_whnp += 1

        if is_span_whtoken(wh_span):
            wh_phrase_is_wh_token += 1

            second_span, secondspan_location = get_second_parent_span(question)
            root = question.depparse_tree
            deproot_idx = question.depparse_tree["token_index"]
            root_token = question.tokens[deproot_idx]
            root_pos = question.pos[deproot_idx]
            deproot_children = get_dependencies(question.depparse_tree, question)

            root_pos_dist[root_pos] += 1

            # if root_pos == "VERB":
            #     if has_dobj(root) and not has_nsubj(root):
            #         wdobj_wo_nsubj += 1
            #     if has_nsubj(root) and not has_dobj(root):
            #         wnsubj_wo_dobj += 1
            #
            #     print(f"{question.question_str}  |||   {text} - {label}")
            #     print(f"{second_span} {get_subspans(second_span, question.conparse_spans)[0:3]}")
            #     print(f"{deproot_idx} {root_token} {root_pos}")
            #     print(deproot_children)
            #     print()

            if root_pos == "AUX":
                # hash_noun_child = False
                # for c in root[dep_children]:
                #     if question.pos[c[dep_token_idx]] == "NOUN":
                #         hash_noun_child = True

                # if has_dobj(root) and not has_nsubj(root):
                #     wdobj_wo_nsubj += 1
                # if has_nsubj(root) and not has_dobj(root):
                #     wnsubj_wo_dobj += 1

                print(f"{question.question_str}  |||   {text} - {label}")
                print(f"{get_subspans(second_span, question.conparse_spans)[0:]}")
                print(f"{deproot_idx} {root_token} {root_pos}")
                print(deproot_children)
                print()


    print("Total question: {}".format(len(qid2question)))
    print(f"Num WHNP: {num_whnp} wh_phrase_is_wh_token: {wh_phrase_is_wh_token}")
    print("-- WHNP-phrase == wh-token -- ")
    print("root_pos_dist: {}".format(root_pos_dist))
    print(" -- -- -- --")



def main(args):

    # squad_json = args.squad_json
    train_or_dev = args.train_dev

    squad_ques_jsonl = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_questions.jsonl"
    squad_ques_conparse_jsonl = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_questions_parse.jsonl"
    squad_ques_depparse_jsonl = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_questions_depparse.jsonl"


    # squad_ques_jsonl = args.squad_ques_jsonl
    # squad_ques_conparse_jsonl = args.squad_ques_conparse_jsonl
    # squad_ques_depparse_jsonl = args.squad_ques_depparse_jsonl

    squad_questions = read_jsonl(squad_ques_jsonl)
    squad_questions_conparse = read_jsonl(squad_ques_conparse_jsonl)
    squad_questions_depparse = read_jsonl(squad_ques_depparse_jsonl)


    qid2question: Dict[str, Question] = map_question_parse(squad_questions=squad_questions,
                                                           squad_questions_conparse=squad_questions_conparse,
                                                           squad_questions_depparse=squad_questions_depparse)

    wh_span_distribution(qid2question)
    # process_WHNP_span(qid2question)

    # squad_dataset = read_json_dataset(squad_json)
    # squad_drop_format = convert_squad_to_drop(squad_dataset=squad_dataset, qid2qinfo=qid2qinfo)
    #
    # output_json = args.output_json
    # print(f"Writing squad drop-formatted data to : {output_json}")
    # with open(output_json, 'w') as outf:
    #     json.dump(squad_drop_format, outf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dev")
    # parser.add_argument("--squad_ques_jsonl")
    # parser.add_argument("--squad_ques_conparse_jsonl")
    # parser.add_argument("--squad_ques_depparse_jsonl")
    args = parser.parse_args()

    main(args)

