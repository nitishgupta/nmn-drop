import os
import json
import random
from semqa.utils.qdmr_utils import read_json_dataset

random.seed(42)


def get_paraid2text(squad_dataset):
    paraid2text = {}
    paraid2qids = {}
    qid2paraquesans = {}
    for article in squad_dataset:
        title = article["title"]
        for para_idx, paragraph_json in enumerate(article["paragraphs"]):
            paraid = f"{title}_{para_idx}"
            paragraph = paragraph_json["context"]
            paraid2text[paraid] = paragraph
            paraid2qids[paraid] = []
            for question_answer in paragraph_json["qas"]:
                qid = question_answer["id"]
                question_text = question_answer["question"].strip().replace("\n", "")
                answer_text = [answer["text"] for answer in question_answer["answers"]][0]
                qid2paraquesans[qid] = (paraid, question_text, answer_text)
                paraid2qids[paraid].append(qid)

    return paraid2text, paraid2qids, qid2paraquesans


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


def write_randomized_output_tsv(paraid2text, paraid2qids, qid2paraquesans, output_tsv, num_ques: int=100):
    print("Writing randomized TSV ...")
    outf = open(output_tsv, 'w')
    tsv_header = "Para-id\tParagraph\tQuestion\tAnswers\n"
    outf.write(tsv_header)

    qids = list(qid2paraquesans.keys())
    random.shuffle(qids)

    for i in range(num_ques):
        qid = qids[i]
        paraid, question, answer = qid2paraquesans[qid]
        para_text = paraid2text[paraid]
        tsv_line = f"{paraid}\t{para_text}\t{question}\t{answer}\n"
        outf.write(tsv_line)


    outf.close()
    print("NumQ: {} written to output TSV: {}".format(num_ques, output_tsv))


def main():
    squad_train_json = "/shared/nitishg/data/squad/squad-train-v1.1.json"
    squad_dev_json = "/shared/nitishg/data/squad/squad-dev-v1.1.json"

    data_dir = os.path.split(squad_train_json)[0]
    train_filename = os.path.split(squad_train_json)[1]
    dev_filename = os.path.split(squad_dev_json)[1]

    analysis_dir = os.path.join(data_dir, "analysis_tsv")
    os.makedirs(analysis_dir, exist_ok=True)

    output_train_tsv = os.path.join(analysis_dir, train_filename[:-4] + "tsv")
    output_dev_tsv = os.path.join(analysis_dir, dev_filename[:-4] + "tsv")

    print("Reading training data ...")
    squad_train = read_json_dataset(squad_train_json)
    squad_train = squad_train["data"]

    print("Reading dev data ...")
    squad_dev = read_json_dataset(squad_dev_json)
    squad_dev = squad_dev["data"]

    # write_output_tsv(squad_train, output_train_tsv)
    # write_output_tsv(squad_dev, output_dev_tsv)

    paraid2text, paraid2qids, qid2paraquesans = get_paraid2text(squad_train)
    write_randomized_output_tsv(paraid2text, paraid2qids, qid2paraquesans, output_train_tsv)




if __name__=="__main__":
    main()
