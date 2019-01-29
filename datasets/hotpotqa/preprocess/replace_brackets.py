import os
import sys
import copy
import time
import json
import argparse
from typing import List, Tuple, Any, Dict

from utils import TAUtils, util, spacyutils
from datasets.hotpotqa.utils import constants


def replaceBrackets(input_jsonl: str, output_jsonl: str) -> None:
    """ Paranthesis in the question cause trouble in logical form execution. Replacing here with -LRB- and -RRB-. """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output jsonl: {}".format(output_jsonl))

    # Reading all objects a priori since the input_jsonl can be overwritten
    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    numdocswritten = 0

    stime = time.time()

    with open(output_jsonl, 'w') as outf:
        for jsonobj in jsonobjs:

            ques = jsonobj[constants.q_field]
            ques = ques.replace("(", constants.LRB)
            ques = ques.replace(")", constants.RRB)
            jsonobj[constants.q_field] = ques

            outf.write(json.dumps(jsonobj))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 1000 == 0:
                ttime = time.time() - stime
                ttime = float(ttime)/60.0
                print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Number of docs written: {}".format(numdocswritten))


def main(args):
    print('Replacing brackets: {}'.format(args.input_jsonl))

    print("Outputting a new file")
    assert args.output_jsonl is not None, "Output_jsonl needed"
    output_jsonl = args.output_jsonl

    # args.input_jsonl --- is the output from preprocess.tokenize
    replaceBrackets(input_jsonl=args.input_jsonl, output_jsonl=output_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl', required=True)
    args = parser.parse_args()

    main(args)