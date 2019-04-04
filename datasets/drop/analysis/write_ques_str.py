import json


infile = "/srv/local/data/nitishg/data/drop/date_subset/drop_dataset_train.json"

outfile = "/srv/local/data/nitishg/data/drop/date_subset/train_ques.txt"


with open(infile) as f:
    dataset = json.load(f)

with open(outfile, 'w') as f:
    for pid, pinfo in dataset.items():
        qapairs = pinfo["qa_pairs"]
        for qapair in qapairs:
            q = qapair["question"]
            f.write(f"{q}\n")

print("Done")
