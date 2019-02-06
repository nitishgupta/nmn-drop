import os
import json
import argparse


from utils import util, spacyutils
from datasets.hotpotqa.utils import constants


def extractBridgeEntityQues(input_jsonl: str, bridgeen_jsonl: str) -> None:
    """ Extract bool questions from a given preprocessed input jsonl that contain the word either/or """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(bridgeen_jsonl))

    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of questions in input: {}".format(len(jsonobjs)))

    outf = open(bridgeen_jsonl, 'w')

    bridge_ques = 0
    bridge_en_ques = 0
    easy_ques = 0
    hard_ques = 0

    for jsonobj in jsonobjs:

        ans_type = jsonobj[constants.ans_type_field]
        level = jsonobj[constants.qlevel_field]
        qtype = jsonobj[constants.qtyte_field]

        if level == 'easy':
            easy_ques += 1
            continue
        hard_ques += 1

        if qtype == 'bridge':
            bridge_ques += 1
            if ans_type != constants.ENTITY_TYPE:
                continue
            else:
                bridge_en_ques += 1
                outf.write(json.dumps(jsonobj))
                outf.write('\n')


    outf.close()

    print(f"Number of easy ques: {easy_ques}")
    print(f"Number of hard ques: {hard_ques}. Bridge: {bridge_ques}. Bridge ques w/ entity ans: {bridge_en_ques}")


def main(args):
    """ Makes an output file in the output_dir as the same name as the input file

        Make sure the input_dir is not the same as the output dir
    """

    print('Extracting either/or questions from : {}'.format(args.input_jsonl))
    dirpath, filename = os.path.split(args.input_jsonl)

    util.makedir(args.output_dir)

    bridgeen_jsonl = os.path.join(args.output_dir, filename)

    # args.input_jsonl --- is the preprocessed jsonl file for a split
    extractBridgeEntityQues(input_jsonl=args.input_jsonl, bridgeen_jsonl=bridgeen_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    main(args)