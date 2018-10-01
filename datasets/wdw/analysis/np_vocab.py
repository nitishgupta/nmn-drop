import os
import json
import argparse

import utils.util as util

nptokens2count = {}

def dataStats(input_jsonl: str) -> None:
    """ Make nptokens2count doct containing all tokens appearing in NP chunks in the documents"""


    print("Reading dataset: {}".format(input_jsonl))
    with open(input_jsonl, 'r') as f:
        lines = f.readlines()
        docDicts = [json.loads(s) for s in lines]


    for i, doc in enumerate(docDicts):
        sentences = doc['contextPara']
        nps = doc['contextPara_NPs']
        for np_sents, sent in zip(nps, sentences):
            for np in np_sents:
                tokens = sent[np[0]:np[1] - 1]
                for t in tokens:
                    t = t.lower()
                    nptokens2count[t] = nptokens2count.get(t, 0) + 1

    print("Size of NP tokens dict: {}".format(len(nptokens2count)))

    print(util.sortDict(nptokens2count, decreasing=False)[0:100])
    print(util.sortDict(nptokens2count, decreasing=True)[0:100])


def read_existing_vocab(vocabpath):
    print("Reading vocab from : {}".format(vocabpath))
    vocab = set(util.readlines(vocabpath))
    return vocab


def get_vocab_intersection(vocab1, vocab2):
    vocab1 = set(vocab1)
    vocab2 = set(vocab2)

    intersection_size = len(vocab1.intersection(vocab2))
    print("Intersection in vocabs: {}".format(intersection_size))


def main(args):
    trainfile = os.path.join(args.tokenized_data, 'train_sorted.jsonl')
    # relax_trainfile = os.path.join(args.tokenized_data, 'train_relax.jsonl')
    # devfile = os.path.join(args.tokenized_data, 'val.jsonl')
    # testfile = os.path.join(args.tokenized_data, 'test.jsonl')

    dataStats(input_jsonl=trainfile)
    # dataStats(input_jsonl=devfile)
    # dataStats(input_jsonl=testfile)

    if args.existing_vocab is not None:
        existing_vocab = read_existing_vocab(args.existing_vocab)
        get_vocab_intersection(existing_vocab, nptokens2count.keys())


if __name__ == '__main__':
    tokenized_data_def = '/save/ngupta19/datasets/WDW/tokenized'

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenized_data', default=tokenized_data_def)
    parser.add_argument('--existing_vocab', default=None)
    args = parser.parse_args()
    main(args)

