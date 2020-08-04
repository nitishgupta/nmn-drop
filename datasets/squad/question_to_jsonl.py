import os
import argparse
import json
from semqa.utils.qdmr_utils import read_json_dataset


def write_output_jsonl(squad_dataset, output_jsonl):
    output_dicts = []
    total_ques = 0

    for article in squad_dataset:
        for para_id, paragraph_json in enumerate(article["paragraphs"]):
            for question_answer in paragraph_json["qas"]:
                total_ques += 1
                question_text = question_answer["question"].strip().replace("\n", "")
                id = question_answer.get("id", None)
                output_dicts.append(
                    {
                        "sentence": question_text,
                        "sentence_id": id,
                    }
                )

    with open(output_jsonl, 'w') as outf:
        for d in output_dicts:
            outf.write(json.dumps(d))
            outf.write("\n")

    print("Written to: {}".format(output_jsonl))


def main(args):
    squad_train_json = args.squad_train_json
    squad_dev_json = args.squad_dev_json

    output_train_jsonl = squad_train_json[:-5] + "_questions.jsonl"
    output_dev_jsonl = squad_dev_json[:-5] + "_questions.jsonl"

    squad_train = read_json_dataset(squad_train_json)
    squad_train = squad_train["data"]

    squad_dev = read_json_dataset(squad_dev_json)
    squad_dev = squad_dev["data"]

    write_output_jsonl(squad_train, output_train_jsonl)
    write_output_jsonl(squad_dev, output_dev_jsonl)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_train_json")
    parser.add_argument("--squad_dev_json")
    args = parser.parse_args()

    main(args)
