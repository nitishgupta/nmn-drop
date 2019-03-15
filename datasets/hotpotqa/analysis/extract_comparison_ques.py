import os
import json
import argparse


from utils import util, spacyutils
from datasets.hotpotqa.utils import constants
from collections import defaultdict

def extractComparisonQues(input_jsonl: str, comparison_jsonl: str) -> None:
    """ Extract bool questions from a given preprocessed input jsonl that contain the word either/or """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(comparison_jsonl))

    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of questions in input: {}".format(len(jsonobjs)))

    outf = open(comparison_jsonl, 'w')

    comparison_ques = 0
    easy_ques = 0
    hard_ques = 0
    hard_comp_ques = 0

    comparison_anstype_dict = defaultdict(int)

    for jsonobj in jsonobjs:

        ans_type = jsonobj[constants.ans_type_field]
        level = jsonobj[constants.qlevel_field]
        qtype = jsonobj[constants.qtyte_field]

        if level == 'easy':
            easy_ques += 1
        else:
            hard_ques += 1

        if qtype == 'comparison':
            comparison_anstype_dict[ans_type] += 1
            if ans_type != constants.ENTITY_TYPE:
                continue
            comparison_ques += 1
            if not level == 'easy':
                hard_comp_ques += 1
            outf.write(json.dumps(jsonobj))
            outf.write('\n')

    outf.close()
    print(comparison_anstype_dict)
    print(f"Number of easy ques: {easy_ques}. Number of hard ques: {hard_ques}")
    print(f"Comparison ques: {comparison_ques}. Hard Comparison questions: {hard_comp_ques}")



def main(args):
    """ Makes an output file in the output_dir as the same name as the input file

        Make sure the input_dir is not the same as the output dir
    """

    print('Extracting comparison questions from: {}'.format(args.input_jsonl))
    dirpath, filename = os.path.split(args.input_jsonl)

    util.makedir(args.output_dir)

    comparison_jsonl = os.path.join(args.output_dir, filename)

    # args.input_jsonl --- is the preprocessed jsonl file for a split
    extractComparisonQues(input_jsonl=args.input_jsonl, comparison_jsonl=comparison_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    main(args)