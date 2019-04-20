import os
import json
import argparse
import datasets.drop.constants as constants


NGRAMS = ["how many field goals did", "how many field goals were",
          "how many yards was the", "how many yards longer was", "how many yards difference",
          "how many points did the", "how many points were",
          "how many years did the", "how many years did it", "how many years after the",
          "how many years passed between", "how many years was",
          "how many touchdowns were scored", "how many touchdowns did the",
          "how many in percent from", "how many in percent of", "how many in percent werent",
          "how many percent of people", "how many percent of the", "how many percent were not",
          "how many touchdown passes did",
          "how many more people are", "how many more yards was",
          "how many total points",
          "how many months did", "how many months after the",
          "how many days did", "how many days after the",
          "how many is the difference",
          "how many passes the", "how many of the",
          "how many times did the", "how many wins did the",
          "how many interceptions did", "how many rushing did", "how many games had the", "how many losses had"]


def get_dataset(infile):
    with open(infile) as f:
        dataset = json.load(f)
    return dataset


def write_ques(train_dataset, ngram: str, ques_filepath: str, qapara_filepath: str):
    qfile = open(ques_filepath, 'w')
    qafile = open(qapara_filepath, 'w')


    for pid, pinfo in train_dataset.items():
        passage = pinfo[constants.cleaned_passage]
        qapairs = pinfo[constants.qa_pairs]

        for qapair in qapairs:
            q = qapair[constants.cleaned_question]
            q_lower = q.lower()
            if constants.answer in qapair:
                answer = qapair[constants.answer]
            else:
                answer = {}

            if constants.answer_passage_spans in qapair:
                ans_as_passage_span = qapair[constants.answer_passage_spans]
            else:
                ans_as_passage_span = 'NONE'

            if ngram in q_lower:
                qfile.write(f"{q}\n")
                qafile.write(f"Questions: {q}\nAnswer: {answer}\nAnsAsPassage: {ans_as_passage_span}\n{passage}\n\n")

    qfile.close()
    qafile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json')
    parser.add_argument('--outputdir')
    args = parser.parse_args()

    train_json = args.train_json
    outputdir_root = args.outputdir

    train_dataset = get_dataset(train_json)

    for ngram in NGRAMS:
        outputdir = os.path.join(outputdir_root, ngram.replace(' ', '_'))
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        ques_file = os.path.join(outputdir, 'train_ques.txt')
        qapara_file = os.path.join(outputdir, 'train_ques_para.txt')
        write_ques(train_dataset, ngram, ques_file, qapara_file)

