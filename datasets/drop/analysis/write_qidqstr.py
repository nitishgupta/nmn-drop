import json
import datasets.drop.constants as constants

dev_infile = "/home1/n/nitishg/code/mhqa/resources/data/drop/preprocess/drop_dataset_dev.json"
dev_outfile = "/home1/n/nitishg/code/mhqa/resources/data/drop/preprocess/dev_q.txt"

def get_dataset(infile):
    with open(infile) as f:
        dataset = json.load(f)
    return dataset


def printDataset(dataset, outfile):
    with open(outfile, 'w') as f:
        for pid, pinfo in dataset.items():
            qapairs = pinfo["qa_pairs"]
            for qapair in qapairs:
                q = qapair[constants.question]
                qid = qapair[constants.query_id]

                f.write("{}\t{}\n".format(qid, q))


dev_dataset = get_dataset(dev_infile)
printDataset(dev_dataset, dev_outfile)
