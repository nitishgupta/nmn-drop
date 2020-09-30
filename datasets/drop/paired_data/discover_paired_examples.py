from typing import List, Tuple, Dict, Set
import os
import json
import copy
import random
import argparse

from collections import defaultdict

from bert_score import BERTScorer

import itertools
from utils.util import tokenize
from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    lisp_to_nested_expression, nested_expression_to_tree, convert_answer
from utils.spacyutils import getSpacyDoc, getSpacyNLP, getNER

spacy_nlp = getSpacyNLP()
scorer = BERTScorer(model_type="bert-base-uncased", rescale_with_baseline=False)

strpair2simscore = defaultdict(float)
strpair2nermatch = defaultdict(float)
str2ners = defaultdict(set)

def get_bertscorer_fscores(cands, refs):
    bert_out = scorer.score(cands, refs)
    precision, recall, fscores = bert_out[0], bert_out[1], bert_out[2]
    return fscores


def get_select_args(node: Node):
    args = set()
    if node.predicate == "select_passage":
        if node.string_arg is not None and node.string_arg != "":
            args.add(node.string_arg)

    for child in node.children:
        args.update(get_select_args(child))

    return args


def get_ners(strarg):
    ners = []
    doc = getSpacyDoc(strarg, spacy_nlp)
    # List of (text, start, end, label)
    ners_mentions = getNER(doc)
    for ner_mention in ners_mentions:
        text = ner_mention[0]
        text = text.replace(" 's ", "")
        text = text.replace(" 's", "")
        text = text.replace(" the ", "")
        text = text.replace("the ", "")
        text = text.replace(" the", "")
        text = text.replace("  ", " ")
        ners.append(text)
    ners = set(ners)
    return ners


def compute_postorder_position(node: Node, position: int = 0):
    for c in node.children:
        position = compute_postorder_position(c, position)
    node.post_order = position
    position += 1
    return position


def make_argpair_sim_matrix(drop_dataset):
    global strpair2simscore
    global strpair2nermatch

    print("Making string-arg pair to F1 and NER-match dictionaries ...")

    paras_done = 0
    for pid, pinfo in drop_dataset.items():
        # Select args for a single paragraph
        select_args = set()
        for qa in pinfo[constants.qa_pairs]:
            if constants.program_supervision not in qa:
                continue

            program_node = node_from_dict(qa[constants.program_supervision])
            select_args.update(get_select_args(program_node))

        # All select-args in all questions in this paragraph
        select_args = list(select_args)
        # Collect all ners in these str-args
        for strarg in select_args:
            if strarg not in str2ners:
                str2ners[strarg] = get_ners(strarg)

        # Make all combination of str-arg pairs to compute similarity; assume metric is symmetric
        refs = []
        cands = []
        for (x, y) in itertools.combinations(select_args, 2):
            refs.append(x)
            cands.append(y)
        if len(cands) > 0 and len(refs) > 0:
            fscores = get_bertscorer_fscores(cands, refs)
            for (x, y, f1) in zip(cands, refs, fscores):
                strpair2simscore[(x, y)] = f1
                strpair2simscore[(y, x)] = f1

                # Compute ner-match between strargs; match IF (1) both NER sets empty OR (2) one is superset of other
                x_ners = str2ners[x]
                y_ners = str2ners[y]
                if len(x_ners) == 0 and len(y_ners) == 0:
                    # match = 1 if both empty
                    ner_match = 1
                elif (len(x_ners) > 0 and len(y_ners) == 0) or (len(x_ners) == 0 and len(y_ners) > 0):
                    # match = 0 if only one is empty
                    ner_match = 0
                else:
                    # match = 1 if either one is a superset
                    ner_match = int(x_ners.issuperset(y_ners) or y_ners.issuperset(x_ners))
                strpair2nermatch[(x, y)] = ner_match
                strpair2nermatch[(y, x)] = ner_match
        paras_done += 1

    if paras_done % 100:
        print(f"Paras done: {paras_done}")


def read_strpair_f1(strpair_f1_tsv):
    global strpair2simscore
    global strpair2nermatch
    print("Reading strpair2simscore from: {}".format(strpair_f1_tsv))
    with open(strpair_f1_tsv, 'r') as f:
        for line in f:
            x, y, f1, nermatch = line.strip().split("\t")
            f1 = float(f1)
            nermatch = float(nermatch)
            strpair2simscore[(x, y)] = f1
            strpair2nermatch[(x, y)] = nermatch
    print("Done!")
    print("Size of strpair2simscore: {}".format(len(strpair2simscore)))
    print("Size of strpair2nermatch: {}".format(len(strpair2nermatch)))


def write_strpair_f1(strpair_f1_tsv):
    global strpair2simscore
    print("Writing strpair2simscore to: {}".format(strpair_f1_tsv))
    with open(strpair_f1_tsv, 'w') as outf:
        for (x, y), f1 in strpair2simscore.items():
            nermatch = strpair2nermatch[(x, y)]
            outf.write(f"{x}\t{y}\t{f1}\t{nermatch}\n")
    print("Done! Size: {}".format(len(strpair2simscore)))


def get_selectarg_and_postorder(node: Node):
    args = []
    if node.predicate == "select_passage":
        if node.string_arg is not None and node.string_arg != "":
            args.append((node.string_arg, node.post_order))
    for child in node.children:
        args.extend(get_selectarg_and_postorder(child))
    return args


def find_paired_examples(drop_dataset):
    global strpair2simscore
    global strpair2nermatch

    print("Finding paired examples within DROp dataset ... ")

    paras_done = 0
    num_ques_w_pairedexamples = 0
    total_paired_examples = 0
    max_paired = 0
    total_qa = 0
    for pid, pinfo in drop_dataset.items():
        paras_done += 1
        if paras_done % 100 == 0:
            print(f"Paras done: {paras_done}")
        # make qid -> List of (select_arg, post-order) tuples list to identify pairing
        qid2select_args = {}
        qid2qadict = {}
        for qa in pinfo[constants.qa_pairs]:
            total_qa += 1
            qid = qa[constants.query_id]
            if constants.program_supervision not in qa:
                continue
            program_node = node_from_dict(qa[constants.program_supervision])
            compute_postorder_position(program_node, 0)

            # List of (select_arg, postorder) tuples for this program
            selectargs_postorders: List[Tuple] = get_selectarg_and_postorder(program_node)
            qid2select_args[qid] = selectargs_postorders
            qid2qadict[qid] = qa

        for qa in pinfo[constants.qa_pairs]:
            paired_examples = []
            qid = qa[constants.query_id]
            orig_question = qa[constants.question]
            if constants.program_supervision not in qa:
                continue
            program_node = node_from_dict(qa[constants.program_supervision])
            orig_lisp = nested_expression_to_lisp(program_node.get_nested_expression())
            compute_postorder_position(program_node, 0)

            # Traverse program-node in-order and figure out closest select-node from other questions
            stack = []  # initialize stack
            stack.append(program_node)
            while len(stack) > 0:
                node = stack.pop()
                best_match = ()
                best_score = 0
                # Process this node
                if node.predicate == "select_passage":
                    if node.string_arg is not None and node.string_arg != "":
                        orig_arg = node.string_arg
                        for paired_qid in qid2select_args:
                            if paired_qid == qid:
                                continue
                            else:
                                for (paired_arg, paired_postorder) in qid2select_args[paired_qid]:
                                    f1 = strpair2simscore[(orig_arg, paired_arg)]
                                    nermatch = strpair2nermatch[(orig_arg, paired_arg)]
                                    if nermatch > 0 and f1 > 0.6 and f1 > best_score:
                                        best_score = f1
                                        best_match = (paired_qid, paired_postorder, node.post_order)

                if best_match != ():
                    (paired_qid, paired_postorder, orig_postorder) = best_match
                    paired_qa: Dict = copy.deepcopy(qid2qadict[paired_qid])
                    paired_qa.pop(constants.shared_substructure_annotations, None)
                    extra_annotation = {
                        "orig_program_lisp": orig_lisp,
                        "orig_question": orig_question,
                        "origprog_postorder_node_idx": orig_postorder,
                        "sharedprog_postorder_node_idx": paired_postorder,
                        # These divnodes are added later; output
                        "origprog_postorder_divnode_idx": -1,
                        "sharedprog_postorder_divnode_idx": -1,
                        # should be different for these nodes
                    }
                    paired_qa.update(extra_annotation)
                    paired_examples.append(paired_qa)
                # end processing this node

                for c in node.children:
                    stack.append(c)
            # end-program-traversal

            if paired_examples:
                num_ques_w_pairedexamples += 1
                total_paired_examples += len(paired_examples)
                max_paired = max(max_paired, len(paired_examples))
                qa[constants.shared_substructure_annotations] = paired_examples

    print()
    print("Total qa: {}".format(total_qa))
    print("Num of question w/ paired examples: {}".format(num_ques_w_pairedexamples))
    print("Total num of paired examples: {}".format(total_paired_examples))
    print("Max num-paired for any question: {}".format(max_paired))
    print()

    stats = {
        "total_qa": total_qa,
        "num_ques_w_pairedexamples": num_ques_w_pairedexamples,
        "total_paired_examples": total_paired_examples,
        "max_paired": max_paired,
    }
    return drop_dataset, stats



def main(args):
    global strpair2simscore
    input_json = args.input_json
    strpair_f1_tsv = args.strpair_f1_tsv

    print("\nDiscovering paired data within DROP ... ")
    print(f"Reading dataset: {input_json}")
    input_dataset = read_drop_dataset(input_json)

    if os.path.exists(strpair_f1_tsv):
        read_strpair_f1(strpair_f1_tsv)
    else:
        make_argpair_sim_matrix(input_dataset)
        write_strpair_f1(strpair_f1_tsv)

    output_dataset, stats_dict = find_paired_examples(drop_dataset=input_dataset)

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
    parser.add_argument("--strpair_f1_tsv")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    main(args)

