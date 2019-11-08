import os
import json

input_dir = "./resources/data/drop_acl/raw"
output_dir = input_dir


def readDataset(input_json):
    with open(input_json, "r") as f:
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
    print(f"Paras sampled: {paras_done}")
    return output_dataset


def write_sample(input_json, output_json, num_paras):
    input_dataset = readDataset(input_json)
    output_dataset = make_sample(input_dataset, num_paras=num_paras)
    with open(output_json, "w") as f:
        json.dump(output_dataset, f, indent=4)


def main():
    train_json = os.path.join(input_dir, "drop_dataset_train.json")
    output_json = os.path.join(output_dir, "sample_train.json")
    write_sample(train_json, output_json, 50)

    dev_json = os.path.join(input_dir, "drop_dataset_dev.json")
    output_json = os.path.join(output_dir, "sample_dev.json")
    write_sample(dev_json, output_json, 50)


if __name__=="__main__":
    main()