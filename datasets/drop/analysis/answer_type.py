import json
import argparse
from collections import defaultdict

from datasets.drop import constants


interesting_tokens = [" the most ", " atleast "]

def answerTypeAnalysis(input_json: str) -> None:
    """ Perform some analysis on answer types. """

    print("Reading input json: {}".format(input_json))

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json, "r") as f:
        dataset = json.load(f)

    print("Number of docs: {}".format(len(dataset)))

    numq = 0
    num_relevant_q = 0


    for pid, pinfo in dataset.items():
        for qa in pinfo[constants.qa_pairs]:
            numq += 1
            question = qa[constants.question]
            if any([x in question for x in interesting_tokens]):
                print(question)
                num_relevant_q += 1


    print(f"Num of Q:{numq}")
    print(f"Num of relevant Q:{num_relevant_q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    args = parser.parse_args()

    answerTypeAnalysis(input_json=args.input_json)
