import os
import json
import string
from allennlp.data.tokenizers import SpacyTokenizer

input_json = "/shared/nitishg/data/squad/squad-train-v1.1_drop.json"
tokenizer = SpacyTokenizer()
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])

def read_json_dataset(input_json: str):
    with open(input_json, "r") as f:
        dataset = json.load(f)
    return dataset


squad_dataset = read_json_dataset(input_json)
numq = 0
for paraid, parainfo in squad_dataset.items():
    for qapair in parainfo["qa_pairs"]:
        answer_dict = qapair["answer"]
        answer_text = answer_dict["spans"][0]
        answer_tokens = tokenizer.tokenize(answer_text)
        answer_text = " ".join(token.text for token in answer_tokens)
        answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        if len(answer_tokens) == 0:
            print(answer_text)
            import pdb
            pdb.set_trace()
        numq += 1

print(numq)
