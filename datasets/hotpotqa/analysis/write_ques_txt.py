import os
import json
import argparse


from utils import util, spacyutils
from datasets.hotpotqa.utils import constants


def writeQuesTxt(input_jsonl: str, outpath_txt: str) -> None:
    """ Extract bool questions from a given preprocessed input jsonl that contain the word either/or """

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(outpath_txt))

    jsonobjs = util.readJsonlDocs(input_jsonl)

    print("Number of questions in input: {}".format(len(jsonobjs)))

    outf = open(outpath_txt, 'w')

    for jsonobj in jsonobjs:

        ans = jsonobj[constants.ans_field]
        ques = jsonobj[constants.q_field]
        contexts = jsonobj[constants.context_field]

        outf.write(f"{ques}\t{ans}\n")

    outf.close()

    print(f"Output written")



def main(args):
    """ Makes an output txt file with one question per line for manual analysis in the same dir as input """

    print('Writing question txt file for: {}'.format(args.input_jsonl))
    dirpath, filename = os.path.split(args.input_jsonl)

    output_filename = filename[:-6] + '_ques.txt'

    outpath_txt = os.path.join(dirpath, output_filename)

    writeQuesTxt(input_jsonl=args.input_jsonl, outpath_txt=outpath_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    args = parser.parse_args()

    main(args)