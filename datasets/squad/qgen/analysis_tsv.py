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
            paired_annotations = qa.get(constants.shared_substructure_annotations, None)
            if paired_annotations:
                paired_annotation = paired_annotations[0]
                paired_question = paired_annotation["question"]
                paired_answer = paired_annotation["answer"]["spans"][0]
                qid2paraquesans[qid] = (passage_id, question_text, answer_text, paired_question, paired_answer)
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
        paraid, question, answer, paired_question, paired_answer = qid2paraquesans[qid]
        para_text = paraid2text[paraid]
        tsv_line = f"{paraid}\t{para_text}\t{question}\t{answer}\t{paired_question}\n"
        outf.write(tsv_line)


    outf.close()
    print("NumQ: {} written to output TSV: {}".format(num_ques, output_tsv))


def randomized_comparison_tsv(paraid2text, paraid2qids, qid2paraquesans_token, qid2paraquesans_dep,
                              output_tsv, num_ques: int=100):
    print("Writing randomized TSV ...")
    outf = open(output_tsv, 'w')
    tsv_header = "Para-id\tParagraph\tQuestion\tAnswer\tToken-based paired-Question\tToken-answer\t" \
                 "Dependency-based paired-question\tDep-answer\n"
    outf.write(tsv_header)

    qids = list(qid2paraquesans_token.keys())
    random.shuffle(qids)

    for i in range(num_ques):
        qid = qids[i]
        paraid, question, answer, token_paired_question, token_ans = qid2paraquesans_token[qid]
        _, _, _, dep_paired_question, dep_ans = qid2paraquesans_dep[qid]
        para_text = paraid2text[paraid]
        tsv_line = f"{paraid}\t{para_text}\t{question}\t{answer}\t{token_paired_question}\t{token_ans}\t" \
                   f"{dep_paired_question}\t{dep_ans}\n"
        outf.write(tsv_line)

    outf.close()
    print("output written to : {}".format(output_tsv))


def token_vs_dep():
    token_squad_train_json = "/shared/nitishg/data/squad/squad-train-v1.1_drop-wcontrastive.json"
    dep_squad_train_json = "/shared/nitishg/data/squad/squad-train-v1.1_drop-wcontrastive_dep.json"

    output_tsv = "/shared/nitishg/data/squad/analysis_tsv/token_vs_dep_contrastive.tsv"

    token_squad_train = read_json_dataset(token_squad_train_json)
    dep_squad_train = read_json_dataset(dep_squad_train_json)

    paraid2text, paraid2qids, token_qid2paraquesans = get_paraid2text(token_squad_train)
    paraid2text, paraid2qids, dep_qid2paraquesans = get_paraid2text(dep_squad_train)

    randomized_comparison_tsv(paraid2text, paraid2qids, token_qid2paraquesans, dep_qid2paraquesans, output_tsv)


def main():
    squad_train_json = "/shared/nitishg/data/squad/squad-train-v1.1_drop-wcontrastive.json"

    data_dir = os.path.split(squad_train_json)[0]
    train_filename = os.path.split(squad_train_json)[1]

    analysis_dir = os.path.join(data_dir, "analysis_tsv")
    os.makedirs(analysis_dir, exist_ok=True)

    token_vs_dep()

    # output_train_tsv = os.path.join(analysis_dir, train_filename[:-4] + "tsv")

    # print("Reading training data ...")
    # squad_train = read_json_dataset(squad_train_json)

    # paraid2text, paraid2qids, qid2paraquesans = get_paraid2text(squad_train)
    # write_randomized_output_tsv(paraid2text, paraid2qids, qid2paraquesans, output_train_tsv)


if __name__=="__main__":
    main()
