import os
import json
import argparse
from typing import List, Tuple

def NPStats(nps: List[Tuple[int]]) -> Tuple[int, int]:
    """ Get number of nps and maxlen of a NP span"""
    num_nps = 0
    maxlen = 0

    for sent in nps:
        for np in sent:
            num_nps += 1
            nplen = (np[1] - np[0])
            maxlen = nplen if maxlen < nplen else maxlen

    return (num_nps, maxlen)


def dataStats(input_jsonl: str, sent_key: str, nps_key: str) -> None:
    print("Reading dataset: {}".format(input_jsonl))

    maxDocSents = -1
    maxSenLength = -1
    numSents = 0
    maxDocWords = 0
    minDocWords = 100000

    longestsentence = []
    smallestDoc = ""

    avgDocWords = 0
    maxDocLenIdx = 0
    minDocLenIdx = 0
    num_NPs = 0
    maxNP_perdoc = 0
    maxLenNP = 0

    docidx = 0

    with open(input_jsonl, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            sentences = doc[sent_key]

            numSents += len(sentences)
            if len(sentences) > maxDocSents:
                maxDocSents = len(sentences)

            sent_lens = [len(s) for s in sentences]



            tokens = [word for sent in sentences for word in sent]
            avgDocWords += len(tokens)
            if len(tokens) > maxDocWords:
                maxDocWords = len(tokens)
                maxDocLenIdx = docidx

            if len(tokens) < minDocWords:
                minDocWords = len(tokens)
                smallestDoc = doc
                minDocLenIdx = docidx

            for i, lens in enumerate(sent_lens):
                if lens > maxSenLength:
                    maxSenLength = lens
                    longestsentence = sentences[i]

            (numnp, maxlen_np) = NPStats(nps=doc[nps_key])
            num_NPs += numnp
            maxLenNP = maxlen_np if maxlen_np > maxLenNP else maxLenNP
            maxNP_perdoc = numnp if numnp > maxNP_perdoc else maxNP_perdoc
            docidx += 1

    avgDocWords = float(avgDocWords)/float(docidx)
    avgNPPerdoc = float(num_NPs)/float(docidx)
    print("MaxDocSents: {}".format(maxDocSents))
    print("maxSenLens: {}".format(maxSenLength))
    print("numSents: {}".format(numSents))
    print("\n")
    print("maxDocWords: {} idx: {}".format(maxDocWords, maxDocLenIdx))
    print("minDocWords: {} idx: {}".format(minDocWords, minDocLenIdx))
    print("avgDocWords: {}".format(avgDocWords))
    # print("Smallest Doc: \n {}".format(smallestDoc))
    print("\n")
    print(f"avgNum NPs per doc: {avgNPPerdoc}")
    print(f"Max num of NP in a doc: {maxNP_perdoc}")
    print(f"totalNum NPs: {num_NPs}")
    print(f"Max Lenof NP: {maxLenNP}")


def main(args):

    dataStats(input_jsonl=args.input_jsonl, sent_key=args.sent_key, nps_key=args.nps_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--sent_key', required=True)
    parser.add_argument('--nps_key', required=True)
    args = parser.parse_args()

    main(args)
