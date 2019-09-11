import json

input_dir = "./resources/data/drop/date_num/date_numcq_hmvy_cnt_relprog_500/"
output_dir = input_dir

def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def make_sample(dataset, num_paras):
    output_dataset = {}

    paras_done = 0
    for pid, pinfo in dataset.items():
        output_dataset[pid] = pinfo
        paras_done += 1
        if paras_done == num_paras:
            break

    print(f"Paras done: {paras_done}")

    return output_dataset


train_dataset = readDataset(input_dir + "drop_dataset_train.json")
output_json = output_dir + "sample.json"

output_dataset = make_sample(train_dataset, num_paras=50)

with open(output_json, 'w') as f:
    json.dump(output_dataset, f, indent=4)
