import os
import json
import argparse


from utils import util, spacyutils
from datasets.hotpotqa.utils import constants


def extractBoolQues(input_jsonl: str, outpath_wsame_jsonl: str,
                    outpath_wosame_jsonl: str, outpath_all_jsonl: str) -> None:
    """ Extract bool questions from a given preprocessed input jsonl """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepaths:\n{}\n{}\n{}".format(outpath_wsame_jsonl, outpath_wosame_jsonl, outpath_all_jsonl))

    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(jsonobjs)))

    outf_wsame = open(outpath_wsame_jsonl, 'w')
    outf_wosame = open(outpath_wosame_jsonl, 'w')
    outf_all = open(outpath_all_jsonl, 'w')

    num_all, num_wsame, num_wosame = 0, 0, 0

    for jsonobj in jsonobjs:

        ans = jsonobj[constants.ans_field]

        if ans in ["yes", "no"]:
            outf_all.write(json.dumps(jsonobj))
            outf_all.write("\n")
            num_all += 1

            ques_tokenized = jsonobj[constants.q_field].split(" ")
            if 'same' in ques_tokenized:
                outf_wsame.write(json.dumps(jsonobj))
                outf_wsame.write('\n')
                num_wsame += 1
            else:
                outf_wosame.write(json.dumps(jsonobj))
                outf_wosame.write('\n')
                num_wosame += 1

    outf_all.close()
    outf_wsame.close()
    outf_wosame.close()

    print("Number of docs written - All: {}  w/ same: {} w/o same: {}".format(num_all, num_wsame, num_wosame))


def main(args):
    print('Extracting bool questions from : {}'.format(args.input_jsonl))
    dirpath, filename = os.path.split(args.input_jsonl)
    outfilename_wsame = filename[0:-6] + "_bool_wsame.jsonl"
    outfilename_wosame = filename[0:-6] + "_bool_wosame.jsonl"
    outfilename_all = filename[0:-6] + "_bool.jsonl"

    outpath_wsame = os.path.join(dirpath, outfilename_wsame)
    outpath_wosame = os.path.join(dirpath, outfilename_wosame)
    outpath_all = os.path.join(dirpath, outfilename_all)

    # args.input_jsonl --- is the output from preprocess.tokenize
    extractBoolQues(input_jsonl=args.input_jsonl, outpath_wsame_jsonl=outpath_wsame,
                    outpath_wosame_jsonl=outpath_wosame, outpath_all_jsonl=outpath_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    args = parser.parse_args()

    main(args)