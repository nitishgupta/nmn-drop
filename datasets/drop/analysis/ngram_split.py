import os
import json
import copy
import argparse
import datasets.drop.constants as constants


NUM_NGRAMS = [
    "how many field goals did",
    "how many field goals were",
    "how many yards was",
    "how many yards longer was",
    "how many yards difference",
    "how many points did the",
    "how many points were",
    "how many years did the",
    "how many years did it",
    "how many years after the",
    "how many years passed between",
    "how many years was",
    "how many touchdowns were scored",
    "how many touchdowns did the",
    "how many in percent from",
    "how many in percent of",
    "how many in percent weren't",
    "how many percent of people",
    "how many percent of the",
    "how many percent were not",
    "how many touchdown passes did",
    "how many more people are",
    "how many more yards was",
    "how many total points",
    "how many months did",
    "how many months after the",
    "how many days did",
    "how many days after the",
    "how many is the difference",
    "how many passes",
    "how many of the",
    "how many times did the",
    "how many wins did the",
    "how many interceptions did",
    "how many rushing",
    "how many games had the",
    "how many losses had",
]


SPAN_NGRAMS = [
    "which team",
    "which happened first",
    "which player scored",
    "which player had",
    "which age group",
    "which group",
    "which event happened",
    "which quarterback threw",
    "what was the",
    "what happened",
    "what event happened",
    "what is the",
    "what did the",
    "what year",
    "who scored the",
    "who scored more",
    "who scored first",
    "who threw the",
    "who caught the",
    "who had",
    "who kicked the",
    "who was the",
    "who did the",
    "who won the",
    "who is the",
    "when did the",
    "when was the",
    "how long was",
    "in which",
    "where which",
    "were there more",
    "were there fewer",
    "did there",
]


def get_dataset(infile):
    with open(infile) as f:
        dataset = json.load(f)
    return dataset


def write_ques(train_dataset, ngram: str, ques_filepath: str, qapara_filepath: str, output_json):
    qfile = open(ques_filepath, "w")
    qafile = open(qapara_filepath, "w")

    output_dataset = {}

    for pid, pinfo in train_dataset.items():
        passage = pinfo[constants.cleaned_passage]
        qapairs = pinfo[constants.qa_pairs]

        relevant_qapairs = []
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
                ans_as_passage_span = "NONE"

            if ngram in q_lower:
                relevant_qapairs.append(qapair)
                qfile.write(f"{q}\n")
                qafile.write(f"Questions: {q}\nAnswer: {answer}\nAnsAsPassage: {ans_as_passage_span}\n{passage}\n\n")
        if relevant_qapairs:
            new_pinfo = copy.deepcopy(pinfo)
            new_pinfo[constants.qa_pairs] = relevant_qapairs
            output_dataset[pid] = new_pinfo

    with open(output_json, "w") as outf:
        json.dump(output_dataset, outf, indent=4)

    qfile.close()
    qafile.close()


def writeNgramOtherQues(
    answer_key: str, ngram_list, train_dataset, ques_filepath: str, qapara_filepath: str, output_json: str
):
    qfile = open(ques_filepath, "w")
    qafile = open(qapara_filepath, "w")

    output_dataset = {}

    for pid, pinfo in train_dataset.items():
        passage = pinfo[constants.cleaned_passage]
        qapairs = pinfo[constants.qa_pairs]

        relevant_qapairs = []
        for qapair in qapairs:
            q = qapair[constants.cleaned_question]
            q_lower = q.lower()
            if constants.answer in qapair:
                answer = qapair[constants.answer]
            else:
                continue

            if answer_key in answer and answer[answer_key]:
                if not any(span in q_lower for span in ngram_list):
                    if constants.answer_passage_spans in qapair:
                        ans_as_passage_span = qapair[constants.answer_passage_spans]
                    else:
                        ans_as_passage_span = "NONE"
                    relevant_qapairs.append(qapair)
                    qfile.write(f"{q}\n")
                    qafile.write(
                        f"Questions: {q}\nAnswer: {answer}\nAnsAsPassage: {ans_as_passage_span}\n{passage}\n\n"
                    )

        if relevant_qapairs:
            new_pinfo = copy.deepcopy(pinfo)
            new_pinfo[constants.qa_pairs] = relevant_qapairs
            output_dataset[pid] = new_pinfo

    with open(output_json, "w") as outf:
        json.dump(output_dataset, outf, indent=4)

    qfile.close()
    qafile.close()


def writeNGramFiles(input_json, outputdir_root, questions_output_filename, quespara_output_filename, output_json):
    dataset = get_dataset(input_json)

    for ngram in NUM_NGRAMS:
        dirname = ngram.replace(" ", "_").replace("'", "")
        num_outputdir = os.path.join(outputdir_root, "num", dirname)
        if not os.path.exists(num_outputdir):
            os.makedirs(num_outputdir, exist_ok=True)
        ques_file = os.path.join(num_outputdir, questions_output_filename)
        qapara_file = os.path.join(num_outputdir, quespara_output_filename)
        output_json_path = os.path.join(num_outputdir, output_json)
        write_ques(dataset, ngram, ques_file, qapara_file, output_json_path)

    numother_outputdir = os.path.join(outputdir_root, "num_other")
    if not os.path.exists(numother_outputdir):
        os.makedirs(numother_outputdir, exist_ok=True)
    ques_file = os.path.join(numother_outputdir, questions_output_filename)
    qapara_file = os.path.join(numother_outputdir, quespara_output_filename)
    output_json_path = os.path.join(numother_outputdir, output_json)
    writeNgramOtherQues("number", NUM_NGRAMS, dataset, ques_file, qapara_file, output_json_path)

    for ngram in SPAN_NGRAMS:
        dirname = ngram.replace(" ", "_").replace("'", "")
        span_outputdir = os.path.join(outputdir_root, "span", dirname)
        if not os.path.exists(span_outputdir):
            os.makedirs(span_outputdir, exist_ok=True)
        ques_file = os.path.join(span_outputdir, questions_output_filename)
        qapara_file = os.path.join(span_outputdir, quespara_output_filename)
        output_json_path = os.path.join(span_outputdir, output_json)
        write_ques(dataset, ngram, ques_file, qapara_file, output_json_path)

    spanother_outputdir = os.path.join(outputdir_root, "span_other")
    if not os.path.exists(spanother_outputdir):
        os.makedirs(spanother_outputdir, exist_ok=True)
    ques_file = os.path.join(spanother_outputdir, questions_output_filename)
    qapara_file = os.path.join(spanother_outputdir, quespara_output_filename)
    output_json_path = os.path.join(spanother_outputdir, output_json)
    writeNgramOtherQues("spans", SPAN_NGRAMS, dataset, ques_file, qapara_file, output_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json")
    parser.add_argument("--dev_json")
    parser.add_argument("--outputdir")
    args = parser.parse_args()

    train_json = args.train_json
    dev_json = args.dev_json
    outputdir_root = args.outputdir

    if not os.path.exists(outputdir_root):
        os.makedirs(outputdir_root, exist_ok=True)

    train_json_filename = "drop_dataset_train.json"
    dev_json_filename = "drop_dataset_dev.json"

    writeNGramFiles(train_json, outputdir_root, "train_ques.txt", "train_para_ques.txt", train_json_filename)
    writeNGramFiles(dev_json, outputdir_root, "dev_ques.txt", "dev_para_ques.txt", dev_json_filename)
