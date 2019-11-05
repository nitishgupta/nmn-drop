from typing import List, Dict
import json
from collections import defaultdict

def read_dataset(input_jsonl):
    with open(input_jsonl) as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data


def count_lfs(data: List[Dict]):
    lf_count = defaultdict(int)
    for instance in data:
        lf_count[instance["type"]] += 1


    return lf_count





if __name__=="__main__":
    predictions_jsonl = "./resources/semqa/checkpoints/drop/merged_data/my1200_full/drop_parser_bert/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_0/S_10/al0.9-composed-num/predictions/drop_dev_preds.jsonl"

    predictions = read_dataset(predictions_jsonl)

    lf_count_dict = count_lfs(predictions)

    for k, v in lf_count_dict.items():
        print(f"{v}\t{k}")