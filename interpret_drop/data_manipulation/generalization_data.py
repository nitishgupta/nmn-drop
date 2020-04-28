import os
import json
import argparse


supervision_keys = [
    "strongly_supervised",
    "program_supervised",
    "execution_supervised",
]


def readDataset(input_json):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


train_programs = ["max_find_qtype", "max_filterfind_qtype",
                  "relocate_maxfind_qtype", "relocate_maxfilterfind_qtype"]

dev_programs = ["min_find_qtype", "min_filterfind_qtype",
                "relocate_minfind_qtype", "relocate_minfilterfind_qtype"]

comparison_programs = ["number_comparison", "date_comparison"]


comparison_train_tokens = ["later", "last", "second"] + ["larger", "more", "largest", "bigger",
                                                         "higher", "highest", "most", "greater"]

comparison_dev_tokens = ["first", "earlier", "forst", "firts"] + ["smaller", "fewer", "lowest", "smallest",
                                                                  "less", "least", "fewest", "lower"]


def get_train_dev_dataset(complete_dataset, split: str):
    dataset = {}
    num_paras = 0
    num_qas = 0

    for para_id, para_info in complete_dataset.items():
        qa_pairs = para_info["qa_pairs"]
        selected_qas = []
        for qa in qa_pairs:
            question_tokens = qa["tokenized_question"].split(" ")
            if "qtype" in qa:
                if split == "train" and qa["qtype"] in train_programs:
                    selected_qas.append(qa)
                if split == "dev" and qa["qtype"] in dev_programs:
                    selected_qas.append(qa)

                if qa["qtype"] in comparison_programs:
                    if any([x in question_tokens for x in comparison_train_tokens]) and split == "train":
                        selected_qas.append(qa)
                    if any([x in question_tokens for x in comparison_dev_tokens]) and split == "dev":
                        selected_qas.append(qa)

        if selected_qas:
            dataset[para_id] = para_info
            dataset[para_id]["qa_pairs"] = selected_qas
            num_paras += 1
            num_qas += len(selected_qas)

    print("Num of paras: {}  Num of qas: {}".format(num_paras, num_qas))
    return dataset


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    train_json = "drop_dataset_train.json"
    dev_json = "drop_dataset_dev.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_tr_dataset = get_train_dev_dataset(train_dataset, split="train")
    new_dev_dataset = get_train_dev_dataset(dev_dataset, split="dev")

    with open(output_trnfp, "w") as f:
        json.dump(new_tr_dataset, f, indent=4)

    with open(output_devfp, "w") as f:
        json.dump(new_dev_dataset, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    main(args)