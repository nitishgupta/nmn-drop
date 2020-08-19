import os
import json
import random
from semqa.utils.qdmr_utils import read_json_dataset
from datasets.drop import constants

random.seed(42)


def get_paraid2text(squad_dataset):
    paraid2text = {}
    paraid2qids = {}
    qid2paraquesans = {}

    for passage_id, passage_info in squad_dataset.items():
        passage = passage_info[constants.passage]
        paraid2text[passage_id] = passage
        paraid2qids[passage_id] = []
        for qa in passage_info[constants.qa_pairs]:
            qid = qa[constants.query_id]
            question_text = qa[constants.question]
            answer_dict = qa[constants.answer]
            answer_text = answer_dict["spans"][0]
            paired_annotation = qa.get(constants.shared_substructure_annotations, None)
            if paired_annotation:
                paired_annotation = paired_annotation[0]
                paired_question = paired_annotation["question"]
                qid2paraquesans[qid] = (passage_id, question_text, answer_text, paired_question)
                paraid2qids[passage_id].append(qid)

    return paraid2text, paraid2qids, qid2paraquesans


def write_randomized_output_tsv(paraid2text, paraid2qids, qid2paraquesans, output_tsv, num_ques: int=100):
    print("Writing randomized TSV ...")
    outf = open(output_tsv, 'w')
    tsv_header = "Para-id\tParagraph\tQuestion\tAnswers\tPaired-Question\n"
    outf.write(tsv_header)

    qids = list(qid2paraquesans.keys())
    random.shuffle(qids)

    for i in range(num_ques):
        qid = qids[i]
        paraid, question, answer, paired_question = qid2paraquesans[qid]
        para_text = paraid2text[paraid]
        tsv_line = f"{paraid}\t{para_text}\t{question}\t{answer}\t{paired_question}\n"
        outf.write(tsv_line)


    outf.close()
    print("NumQ: {} written to output TSV: {}".format(num_ques, output_tsv))


def main():
    squad_train_json = "/shared/nitishg/data/squad/wh-phrase/squad-train-v1.1_drop-wcontrastive.json"

    data_dir = os.path.split(squad_train_json)[0]
    train_filename = os.path.split(squad_train_json)[1]

    analysis_dir = os.path.join(data_dir, "analysis_tsv")
    os.makedirs(analysis_dir, exist_ok=True)

    output_train_tsv = os.path.join(analysis_dir, train_filename[:-4] + "tsv")

    print("Reading training data ...")
    squad_train = read_json_dataset(squad_train_json)

    paraid2text, paraid2qids, qid2paraquesans = get_paraid2text(squad_train)
    write_randomized_output_tsv(paraid2text, paraid2qids, qid2paraquesans, output_train_tsv)


if __name__=="__main__":
    main()
