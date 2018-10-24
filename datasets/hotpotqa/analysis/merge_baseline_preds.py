import sys
import ujson as json

import argparse
from typing import List, Tuple, Any, Dict
from datasets.hotpotqa.utils import constants
from utils import util
import datasets.hotpotqa.analysis.hotpot_evaluate_v1 as evaluation

"""
For finding the maximum performance possible by predicting only NER spans as answers
"""

ner_type: List[Any]


def mergepreds(data_jsonl: str, pred_json: str) -> None:
    print("Reading dataset: {}".format(data_jsonl))
    print("Reading predictions: {}".format(pred_json))

    # The pred_json from the baseline model is a json with 'answer' key and value as a answer_dict
    # This answer_dict contains {qid: ans} key, val pairs
    with open(pred_json, 'r') as f:
        pred_obj = json.load(f)
    pred_answers = pred_obj['answer']

    print(f"Number of answers: {len(pred_answers)}")

    qa_examples: List[Dict] = util.readJsonlDocs(data_jsonl)

    print(f"Number of dev examples: {len(qa_examples)}")

    with open(data_jsonl, 'w') as f:
        for qaexample in qa_examples:
            qid = qaexample[constants.id_field]
            if qid in pred_answers:
                predans = pred_answers[qid]
                qaexample[constants.pred_ans] = predans
                f.write(json.dumps(qaexample))
                f.write("\n")

    print("Merged answers")


def main(args):
    mergepreds(data_jsonl=args.data_jsonl, pred_json=args.pred_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_jsonl', required=True)
    parser.add_argument('--pred_json', required=True)
    args = parser.parse_args()

    main(args)

