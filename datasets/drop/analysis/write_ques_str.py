import json
import datasets.drop.constants as constants


input_dir = "date_subset_prune"
train_infile = f"/srv/local/data/nitishg/data/drop_old/{input_dir}/drop_dataset_train.json"
train_outfile = f"/srv/local/data/nitishg/data/drop_old/{input_dir}/train_qa.txt"

dev_infile = f"/srv/local/data/nitishg/data/drop_old/{input_dir}/drop_dataset_dev.json"
dev_outfile = f"/srv/local/data/nitishg/data/drop_old/{input_dir}/dev_qa.txt"


def get_dataset(infile):
    with open(infile) as f:
        dataset = json.load(f)
    return dataset


def printDataset(dataset, outfile):
    with open(outfile, "w") as f:
        for pid, pinfo in dataset.items():
            passage = pinfo[constants.original_passage]
            qapairs = pinfo["qa_pairs"]
            dates = pinfo[constants.passage_date_normalized_values]
            for qapair in qapairs:
                q = qapair[constants.original_question]
                if constants.qspan_dategrounding_supervision in qapair:
                    passage_event_date_groundings = qapair[constants.qspan_dategrounding_supervision]
                else:
                    passage_event_date_groundings = []
                if constants.question_event_date_values in qapair:
                    passage_event_date_values = qapair[constants.question_event_date_values]
                else:
                    passage_event_date_values = []

                answer = qapair[constants.answer]
                f.write(f"{q}\n")
                f.write(f"{passage}\n")
                f.write(f"{dates}\n")
                f.write(f"{answer}\n")
                f.write(f"event_date_values: {passage_event_date_values}")
                f.write(f"event_date_groundings: {passage_event_date_groundings}")


train_dataset = get_dataset(train_infile)
printDataset(train_dataset, train_outfile)

dev_dataset = get_dataset(dev_infile)
printDataset(dev_dataset, dev_outfile)
