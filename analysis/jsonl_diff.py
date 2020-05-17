from typing import List, Dict
import json


jsonl1 = "/shared/nitishg/data/drop-w-qdmr/hmyw-old/train.jsonl"
jsonl2 = "/shared/nitishg/data/drop-w-qdmr/hmyw-old-alt/train.jsonl"


def read_jsonl(input_jsonl) -> List[Dict]:
    with open(input_jsonl, 'r') as f:
        dicts = [json.loads(line) for line in f.readlines()]
    return dicts


def extra_questions(examples_1, examples_2):
    qids1 = [ex["query_id"] for ex in examples_1]
    qids2 = [ex["query_id"] for ex in examples_2]

    qids1_exclusive = set(qids1).difference(set(qids2))

    for ex in examples_1:
        if ex["query_id"] in qids1_exclusive:
            print(ex["question"])


examples_1 = read_jsonl(jsonl1)
examples_2 = read_jsonl(jsonl2)

extra_questions(examples_1, examples_2)
