import json
import datasets.drop.constants as constants

input_dir = "./resources/data/drop_s/date_num/date_numcq_hmvy_cnt_filter_500/"
output_dir = "./resources/data/drop_s/sample/"

def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def make_sample(dataset):
    output_dataset = {}

    paras_done = 0
    ques_done = 0
    for pid, pinfo in dataset.items():
        qas = pinfo[constants.qa_pairs]
        for qa in qas:
            qa[constants.program_supervised] = False
            if constants.qtype in qa:
                qa.pop(constants.qtype)
            qa[constants.qattn_supervised] = False
            qa[constants.exection_supervised] = False
            qa[constants.strongly_supervised] = False

        ques_done += len(qas)

        output_dataset[pid] = pinfo
        paras_done += 1
        if paras_done == 5:
            break

    print(f"Paras done: {paras_done}  Ques: {ques_done}")

    return output_dataset


train_dataset = readDataset(input_dir + "drop_dataset_train.json")
output_json = output_dir + "drop_dataset_train.json"

output_dataset = make_sample(train_dataset)

with open(output_json, 'w') as f:
    json.dump(output_dataset, f, indent=4)
