import os
import json
import copy
import argparse
import datasets.drop.constants as constants


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def quesParaSize(input_json):
    dataset = readDataset(input_json)

    numparas = 0
    numques = 0
    maxparalen = 0
    maxqueslen = 0

    for pid, pinfo in dataset.items():
        numparas += 1
        passage = pinfo[constants.tokenized_passage]
        plen = len(passage.split(' '))
        maxparalen = plen if plen > maxparalen else maxparalen

        qa_pairs = pinfo[constants.qa_pairs]

        for qa in qa_pairs:
            numques += 1
            qlen = len(qa[constants.tokenized_question])
            maxqueslen = qlen if qlen > maxqueslen else maxqueslen



    print(f"Paras: {numparas}  MaxParaLen:{maxparalen}")
    print(f"Questions: {numques}  MaxQuesLen:{maxqueslen}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir')
    args = parser.parse_args()

    inputdir = args.inputdir

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

    inputdir = "./resources/data/drop/date_num/dc_nc_yeardiff"

    input_trnfp = os.path.join(inputdir, train_json)
    input_devfp = os.path.join(inputdir, dev_json)

    print(input_trnfp)
    quesParaSize(input_trnfp)

    print(input_devfp)
    quesParaSize(input_devfp)



