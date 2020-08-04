import os
import json
from semqa.utils.qdmr_utils import read_json_dataset


def write_output_tsv(squad_dataset, output_tsv):
    outf = open(output_tsv, 'w')
    tsv_header = "Title\tPara-id\tParagraph\tQuestion\tAnswers\n"
    outf.write(tsv_header)

    total_ques = 0
    total_para = 0
    total_articles = 0

    for article in squad_dataset:
        title = article["title"]
        total_articles += 1
        for para_id, paragraph_json in enumerate(article["paragraphs"]):
            total_para += 1
            paragraph = paragraph_json["context"]
            for question_answer in paragraph_json["qas"]:
                total_ques += 1
                question_text = question_answer["question"].strip().replace("\n", "")
                answer_texts = [answer["text"] for answer in question_answer["answers"]]
                id = question_answer.get("id", None)
                tsv_line = f"{title}\t{total_para}\t{paragraph}\t{question_text}\t{answer_texts}\n"
                outf.write(tsv_line)
    outf.close()
    print("Output TSV: {}".format(output_tsv))
    print("num_articles: {}  num_paras: {}  num_ques: {}".format(total_articles, total_para, total_ques))


def main():
    squad_train_json = "/shared/nitishg/data/squad/squad-train-v1.1.json"
    squad_dev_json = "/shared/nitishg/data/squad/squad-dev-v1.1.json"

    output_train_tsv = squad_train_json[:-4] + "tsv"
    output_dev_tsv = squad_dev_json[:-4] + "tsv"

    squad_train = read_json_dataset(squad_train_json)
    squad_train = squad_train["data"]

    squad_dev = read_json_dataset(squad_dev_json)
    squad_dev = squad_dev["data"]

    write_output_tsv(squad_train, output_train_tsv)
    write_output_tsv(squad_dev, output_dev_tsv)


if __name__=="__main__":
    main()
