import os
import json
import argparse
from typing import List, Tuple, Any
from datasets.hotpotqa.utils import constants
from utils import util

ner_type: List[Any]


def printQuestions(input_jsonl: str) -> None:
    print("Reading dataset: {}".format(input_jsonl))
    qa_examples = util.readJsonlDocs(input_jsonl)

    toPrint = 200

    for qaexample in qa_examples:

        q = qaexample[constants.q_field]
        qtype = qaexample[constants.qtyte_field]
        ans = qaexample[constants.ans_field]
        qners = qaexample[constants.q_ner_field]

        print(f"[{qtype}]\nQ:{q}\n{ans}")
        print()
        toPrint -= 1
        if toPrint < 0:
            break


def main(args):

    printQuestions(input_jsonl=args.input_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    args = parser.parse_args()

    main(args)
