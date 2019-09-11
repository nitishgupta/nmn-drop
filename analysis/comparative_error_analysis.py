import os

import argparse


def read_prediction_file(file_path):
    """Input files are instance-per-line w/ each line being tab-separated."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    instances = [line.strip().split('\t') for line in lines]

    return instances


def make_correct_incorrect_dicts(instances):
    """Make separate sets of question ids for correctly and in-correctly answered questions.

    Args:
        instances: is a 3(4)-tuple with (q-id, ques-text, 'C'/'NC', logical_form (optional) )

    Returns:
        correct_qids: Set of qids answered correctly
        incorrect_qids: Set of qids answered incorrectly
    """
    correct_qids = set()
    incorrect_qids = set()

    for instance in instances:
        if instance[2] == 'C':
            correct_qids.add(instance[0])
        else:
            incorrect_qids.add(instance[0])

    return correct_qids, incorrect_qids


def make_qid2questiondict(instances):
    """Make question_id to question text map

    Args:
        instances: is a 3(4)-tuple with (q-id, ques-text, 'C'/'NC', logical_form (optional) )

    Returns:
        qid2qtext: Dict from qid 2 qtext
    """
    qid2qtext = {}

    for instance in instances:
        qid = instance[0]
        qtext = instance[1]
        qid2qtext[qid] = qtext

    return qid2qtext


def make_qid2logicalform(instances):
    """Make question_id to predicted logical_form map

    Args:
        instances: is a 3(4)-tuple with (q-id, ques-text, 'C'/'NC', logical_form (optional) )

    Returns:
        qid2lf: Dict from qid 2 logical form
    """
    qid2lf = {}

    for instance in instances:
        qid = instance[0]
        if len(instance) == 4:
            lf = instance[3]
        else:
            lf = ''
        qid2lf[qid] = lf

    return qid2lf


def print_qtexts(qids, qid2qtext):
    for i, qid in enumerate(qids):
        print("{}: {}".format(i, qid2qtext[qid]))


def print_qtext_lf(qids, qid2qtext, qid2lf):
    for i, qid in enumerate(qids):
        print("{}: {} -- {}".format(i, qid2qtext[qid], qid2lf[qid]))


def print_qtext_bothlfs(qids, qid2qtext, qid2lf1, qid2lf2):
    for i, qid in enumerate(qids):
        print("{}: {}\n{}\n{}\n".format(i, qid2qtext[qid], qid2lf1[qid], qid2lf2[qid]))


def diff_in_lfs(qids, qid2lf1, qid2lf2):
    num_diff_lf = 0
    for i, qid in enumerate(qids):
        lf1 = qid2lf1[qid]
        lf2 = qid2lf2[qid]

        if lf1 != lf2:
            num_diff_lf += 1

    return num_diff_lf


def overlap_in_correct_incorrect_questext(correct_qids, incorrect_qids, qid2qtext):
    correct_qtexts = set([qid2qtext[qid] for qid in correct_qids])
    incorrect_qtexts = set([qid2qtext[qid] for qid in incorrect_qids])

    overlap = correct_qtexts.intersection(incorrect_qtexts)

    print("Unique correct questions: {}".format(len(correct_qtexts)))
    print("Unique incorrect questions: {}".format(len(incorrect_qtexts)))
    print("Unique question overlap: {}".format(len(overlap)))


def error_overlap_statistics(file1, file2):
    """Perform analysis on predictions of two different models and see the level of overlap in their predictions.


    The two files are tab-separated, with the form
        Question-id    Question-text    'C' or 'NC'     LogicalForm
    The logical form is only for predictions from our models
    """
    instances1 = read_prediction_file(file1)
    instances2 = read_prediction_file(file2)

    qid2qtext1 = make_qid2questiondict(instances1)
    qid2lf1 = make_qid2logicalform(instances1)

    qid2qtext2 = make_qid2questiondict(instances2)
    qid2lf2 = make_qid2logicalform(instances2)

    num_instances1 = len(instances1)
    num_instances2 = len(instances2)

    correct_qids1, incorrect_qids1 = make_correct_incorrect_dicts(instances1)
    correct_qids2, incorrect_qids2 = make_correct_incorrect_dicts(instances2)

    perf1 = len(correct_qids1)/float(num_instances1)
    perf2 = len(correct_qids2) / float(num_instances2)

    correct_overlap_qids = correct_qids1.intersection(correct_qids2)

    correct1_incorrect2_qids = correct_qids1.intersection(incorrect_qids2)
    incorrect1_correct2_qids = incorrect_qids1.intersection(correct_qids2)

    # For questions predicted correct by M1, and incorrect by M2 -- the number of ques with diff LFs
    c1_nc2_lf_diff = diff_in_lfs(correct1_incorrect2_qids, qid2lf1, qid2lf2)
    nc1_c2_lf_diff = diff_in_lfs(incorrect1_correct2_qids, qid2lf1, qid2lf2)

    print("Correct in Model 1, Incorrect in Model 2")
    print_qtext_bothlfs(correct1_incorrect2_qids, qid2qtext2, qid2lf1, qid2lf2)

    print("Incorrect in Model 1, Correct in Model 2")
    print_qtext_bothlfs(incorrect1_correct2_qids, qid2qtext1, qid2lf1, qid2lf2)


    print("Model 1")
    print("Num instances: {} Correct: {} Incorrect: {} Perf: {}".format(num_instances1, len(correct_qids1),
                                                                        len(incorrect_qids1), perf1))
    print()
    print("Model 2")
    print("Num instances: {} Correct: {} Incorrect: {} Perf: {}".format(num_instances2, len(correct_qids2),
                                                                        len(incorrect_qids2), perf2))

    print("Correct Overlap : {}".format(len(correct_overlap_qids)))

    print("Correct in M1; Incorrect in M2: {}".format(len(correct1_incorrect2_qids)))
    print("Num of ques w/ different LF predictions: {}".format(c1_nc2_lf_diff))
    print()

    print("Incorrect in M1; Correct in M2: {}".format(len(incorrect1_correct2_qids)))
    print("Num of ques w/ different LF predictions: {}".format(nc1_c2_lf_diff))

    print("Model1")
    overlap_in_correct_incorrect_questext(correct_qids1, incorrect_qids1, qid2qtext1)

    print("Model2")
    overlap_in_correct_incorrect_questext(correct_qids2, incorrect_qids2, qid2qtext2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1')
    parser.add_argument('--file2')
    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2


    OurGRUmodel = "/scratch1/nitishg/semqa/checkpoints/drop/date_num/date_numcq_hmvy_cnt_relprog_500/drop_parser/" \
                  "TOKENS_qanet/ED_100/RG_1e-07/MODELTYPE_encoded/CNTFIX_false/aux_true/SUPEPOCHS_5/" \
                  "S_10/CModelBM1" + "/predictions"

    OurBERTmodel = "/scratch1/nitishg/semqa/checkpoints/drop/date_num/date_numcq_hmvy_cnt_relprog_500/" \
                   "drop_parser_bert/CNTFIX_false/aux_true/SUPEPOCHS_5/" \
                   "S_10/BertModelRelAux15" + "/predictions"

    NABERTmodel = ("/home1/n/nitishg/code/drop-bert/checkpoints/nabert-plus-template/mydata/predictions")


    numcomp = "numcomp_full_dev_analysis.tsv"
    relocate = "relocate_wprog_dev_analysis.tsv"
    count = "count_filterqattn_dev_analysis.tsv"
    year_diff = "year_diff_dev_analysis.tsv"

    ###
    model1 = OurBERTmodel
    model2 = NABERTmodel
    dataset = count

    file1 = os.path.join(model1, dataset)

    file2 = os.path.join(model2, dataset)

    assert os.path.exists(file1)
    assert os.path.exists(file2)

    error_overlap_statistics(file1, file2)
