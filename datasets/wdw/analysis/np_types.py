import os
import json
import argparse

import utils.util as util
from typing import Dict, List

nptokens2count = {}


def _tokens_pos(doc: Dict) -> List[List[str]]:
    """ doc: json dict read from file
    Returns:
        List of list of token_pos
    """
    sentences = doc['contextPara']
    nps = doc['contextPara_NPs']
    sents_pos = doc['contextPara_POS']

    token_pos_doc = []
    for s, pos in zip(sentences, sents_pos):
        token_pos_sent = [(t + "_" + p) for t, p in zip(s, pos)]
        token_pos_doc.append(token_pos_sent)

    return token_pos_doc


def getNPsInDoc(jsonline: str, limit: int = None) -> List[str]:
    """ Return all NPs in the document as a list of str.
        Each idx contains token_pos """

    doc = json.loads(jsonline)
    tokens_pos = _tokens_pos(doc)

    nps_doc = doc['contextPara_NPs']

    count = 0

    nps_tokens_w_pos = []

    for i, nps in enumerate(nps_doc):
        for np in nps:
            nps_tokens_w_pos.append(' '.join(tokens_pos[i][np[0]:np[1]]))
            count += 1
            if limit is not None:
                if count == limit:
                    break

    return nps_tokens_w_pos


def printAllDocNPs(input_jsonl: str) -> None:
    NPs = set()
    with open(input_jsonl, 'r') as f:
        line = f.readline()
        while line:
            nps = getNPsInDoc(jsonline=line, limit=None)
            NPs.update(set(nps))
            line = f.readline()

    for np in NPs:
        print(np)



def main(args):
    trainfile = os.path.join(args.tokenized_data, 'train_wpos.jsonl')
    # relax_trainfile = os.path.join(args.tokenized_data, 'train_relax.jsonl')
    # devfile = os.path.join(args.tokenized_data, 'val.jsonl')
    # testfile = os.path.join(args.tokenized_data, 'test.jsonl')

    printAllDocNPs(input_jsonl=trainfile)
    # dataStats(input_jsonl=devfile)
    # dataStats(input_jsonl=testfile)


if __name__ == '__main__':
    tokenized_data_def = '/save/ngupta19/datasets/WDW/tokenized'

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenized_data', default=tokenized_data_def)
    args = parser.parse_args()

    main(args)
