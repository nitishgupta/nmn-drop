from typing import List, Tuple, Dict, Union, Callable
import os
import re
import json
import copy
import argparse
from collections import defaultdict

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    read_json_dataset, read_jsonl
import datasets.drop.constants as constants

from datasets.squad.squad_utils import Question, map_question_parse, get_wh_phrase, is_firstspan_whphrase


def only_whphrase_questions(squad_dataset: Dict, qid2question: Dict[str, Question]):
    """Prune Squad dataset (in DROP format) to only questions that start with a WH-phrase."""

    numq, selected_q = 0, 0
    nump, selected_p = 0, 0

    pruned_dataset = {}

    for passage_id, passage_info in squad_dataset.items():
        nump += 1
        pruned_qa = []

        for qa in passage_info[constants.qa_pairs]:
            numq += 1
            qid = qa[constants.query_id]
            qobj: Question = qid2question[qid]

            if is_firstspan_whphrase(qobj):
                selected_q += 1
                pruned_qa.append(qa)

        if pruned_qa:
            selected_p += 1
            passage_info[constants.qa_pairs] = pruned_qa
            pruned_dataset[passage_id] = passage_info

    print("Original P:{} Q:{}".format(nump, numq))
    print("Pruned P:{} Q:{}".format(selected_p, selected_q))

    return pruned_dataset


def get_qid2question(train_or_dev):
    squad_ques_jsonl = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_questions.jsonl"
    squad_ques_conparse_jsonl = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_questions_parse.jsonl"
    squad_ques_depparse_jsonl = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_questions_depparse.jsonl"

    squad_questions = read_jsonl(squad_ques_jsonl)
    squad_questions_conparse = read_jsonl(squad_ques_conparse_jsonl)
    squad_questions_depparse = read_jsonl(squad_ques_depparse_jsonl)

    qid2question = map_question_parse(squad_questions, squad_questions_conparse, squad_questions_depparse)

    print("Num of questions in {}: {}".format(train_or_dev, len(qid2question)))

    return qid2question


def main(args):

    print("\nPruning SQuAD data to keep questions starting in a WH-Phrase")


    squad_train_json = f"/shared/nitishg/data/squad/squad-train-v1.1_drop.json"
    squad_dev_json = f"/shared/nitishg/data/squad/squad-dev-v1.1_drop.json"

    output_train_json = f"/shared/nitishg/data/squad/wh-phrase/squad-train-v1.1_drop.json"
    output_dev_json = f"/shared/nitishg/data/squad/wh-phrase/squad-dev-v1.1_drop.json"

    print("\nPruning train dataset ...")
    pruned_train_dataset = only_whphrase_questions(read_drop_dataset(squad_train_json),
                                                   get_qid2question(train_or_dev="train"))

    with open(output_train_json, 'w') as outf:
        json.dump(pruned_train_dataset, outf, indent=4)

    print("\nPruning dev dataset ...")
    pruned_dev_dataset = only_whphrase_questions(read_drop_dataset(squad_dev_json),
                                                 get_qid2question(train_or_dev="dev"))

    with open(output_dev_json, 'w') as outf:
        json.dump(pruned_dev_dataset, outf, indent=4)


if __name__=="__main__":
    main(None)






