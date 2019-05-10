import os
import json
import copy
import argparse
import datasets.drop.constants as constants
from collections import defaultdict
from utils.util import round_all


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def quesParaSize(input_json):
    dataset = readDataset(input_json)
    numparas = 0
    maxparalen = 0
    passage_len_sums = 0
    plen_lt_100_cnt = 0
    plen_lt_200_cnt = 0
    plen_lt_400_cnt = 0
    plen_lt_500_cnt = 0
    plen_lt_600_cnt = 0
    plen_lt_800_cnt = 0
    plen_lt_1000_cnt = 0

    for pid, pinfo in dataset.items():
        numparas += 1
        passage = pinfo[constants.tokenized_passage]
        plen = len(passage.split(' '))
        maxparalen = plen if plen > maxparalen else maxparalen

        passage_len_sums += plen

        if plen < 100:
            plen_lt_100_cnt += 1
        if plen < 200:
            plen_lt_200_cnt += 1
        if plen < 400:
            plen_lt_400_cnt += 1
        if plen < 500:
            plen_lt_500_cnt += 1
        if plen < 600:
            plen_lt_600_cnt += 1
        if plen < 800:
            plen_lt_800_cnt += 1
        if plen < 1000:
            plen_lt_1000_cnt += 1

    avg_plen = float(passage_len_sums)/numparas

    print(f"Paras: {numparas}  MaxParaLen:{maxparalen}")
    print(f"Avg Para len: {avg_plen}")
    print(f"Plen < 100: {plen_lt_100_cnt}")
    print(f"Plen < 200: {plen_lt_200_cnt}")
    print(f"Plen < 400: {plen_lt_400_cnt}")
    print(f"Plen < 500: {plen_lt_500_cnt}")
    print(f"Plen < 600: {plen_lt_600_cnt}")
    print(f"Plen < 800: {plen_lt_800_cnt}")
    print(f"Plen < 1000: {plen_lt_1000_cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir')
    args = parser.parse_args()

    inputdir = args.inputdir

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

    inputdir = "./resources/data/drop_s/num/count_filterqattn"
    # inputdir = "./resources/data/drop_s/date_num/date_numcq_hmvy_cnt_filter"


    input_trnfp = os.path.join(inputdir, train_json)
    input_devfp = os.path.join(inputdir, dev_json)

    print(input_trnfp)
    quesParaSize(input_trnfp)

    print(input_devfp)
    quesParaSize(input_devfp)



