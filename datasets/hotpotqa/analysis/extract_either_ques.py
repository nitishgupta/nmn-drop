import os
import json
import argparse


from utils import util, spacyutils
from datasets.hotpotqa.utils import constants


def extractEitherOrQues(input_jsonl: str, outpath_either_jsonl: str) -> None:
    """ Extract bool questions from a given preprocessed input jsonl that contain the word either/or """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(outpath_either_jsonl))

    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of questions in input: {}".format(len(jsonobjs)))

    outf_eitherques = open(outpath_either_jsonl, 'w')

    total_bool_ques = 0
    num_either_ques = 0
    num_yes_ans, num_no_ans = 0, 0

    for jsonobj in jsonobjs:

        ans = jsonobj[constants.ans_field]

        if ans in ["yes", "no"]:
            total_bool_ques += 1
            ques_tokenized = jsonobj[constants.q_field].split(" ")
            if 'either' in ques_tokenized or 'or' in ques_tokenized:
                outf_eitherques.write(json.dumps(jsonobj))
                outf_eitherques.write('\n')
                num_either_ques += 1
                num_yes_ans += 1 if ans == "yes" else 0
                num_no_ans += 1 if ans == "no" else 0

    outf_eitherques.close()

    print(f"Number of bool ques in input: {total_bool_ques}")
    print(f"Number of either/or questions in output: {num_either_ques}. Yes:{num_yes_ans}. No: {num_no_ans}")



def main(args):
    """ Makes an output file in the output_dir as the same name as the input file

        Make sure the input_dir is not the same as the output dir
    """

    print('Extracting either/or questions from : {}'.format(args.input_jsonl))
    dirpath, filename = os.path.split(args.input_jsonl)

    util.makedir(args.output_dir)

    outpath_eitherques = os.path.join(args.output_dir, filename)

    # args.input_jsonl --- is the preprocessed jsonl file for a split
    extractEitherOrQues(input_jsonl=args.input_jsonl, outpath_either_jsonl=outpath_eitherques)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    main(args)