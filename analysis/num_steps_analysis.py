import os
from collections import defaultdict

import argparse

def read_numsteps_annotation(filepath='analysis/dev_numsteps_annotation.tsv'):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    instances = [line.strip().split('\t') for line in lines]

    qid2steps = {}
    for instance in instances:
        qid = instance[0]
        steps = instance[2]
        qid2steps[qid] = int(steps)

    return qid2steps


def read_qid2prediction(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    instances = [line.strip().split('\t') for line in lines]

    qid2pred = {}
    for instance in instances:
        qid = instance[0]
        pred = 1 if instance[2] == 'C' else 0
        qid2pred[qid] = pred

    return qid2pred


def count_step2correct(qid2steps, qid2pred):
    steps2total = defaultdict(float)
    steps2correct = defaultdict(int)
    steps2incorrect = defaultdict(float)

    steps2crrectratio = defaultdict(float)

    for qid, steps in qid2steps.items():
        if qid in qid2pred:
            steps2total[steps] += 1
            steps2correct[steps] += qid2pred[qid]

    for steps, total in steps2total.items():
        correct = steps2correct[steps]

        ration = float(correct)/total
        steps2crrectratio[steps] = ration

    print(steps2total)
    print(steps2correct)
    print(steps2crrectratio)
    return steps2crrectratio




# def read_prediction_file(file_path):
#     """Input files are instance-per-line w/ each line being tab-separated."""
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#
#     instances = [line.strip().split('\t') for line in lines]
#
#     return instances
#
#
# def make_correct_incorrect_dicts(instances):
#     """Make separate sets of question ids for correctly and in-correctly answered questions.
#
#     Args:
#         instances: is a 3(4)-tuple with (q-id, ques-text, 'C'/'NC', logical_form (optional) )
#
#     Returns:
#         correct_qids: Set of qids answered correctly
#         incorrect_qids: Set of qids answered incorrectly
#     """
#     correct_qids = set()
#     incorrect_qids = set()
#
#     for instance in instances:
#         if instance[2] == 'C':
#             correct_qids.add(instance[0])
#         else:
#             incorrect_qids.add(instance[0])
#
#     return correct_qids, incorrect_qids
#
#
# def make_qid2questiondict(instances):
#     """Make question_id to question text map
#
#     Args:
#         instances: is a 3(4)-tuple with (q-id, ques-text, 'C'/'NC', logical_form (optional) )
#
#     Returns:
#         qid2qtext: Dict from qid 2 qtext
#     """
#     qid2qtext = {}
#
#     for instance in instances:
#         qid = instance[0]
#         qtext = instance[1]
#         qid2qtext[qid] = qtext
#
#     return qid2qtext
#
#
# def make_qid2logicalform(instances):
#     """Make question_id to predicted logical_form map
#
#     Args:
#         instances: is a 3(4)-tuple with (q-id, ques-text, 'C'/'NC', logical_form (optional) )
#
#     Returns:
#         qid2lf: Dict from qid 2 logical form
#     """
#     qid2lf = {}
#
#     for instance in instances:
#         qid = instance[0]
#         if len(instance) == 4:
#             lf = instance[3]
#         else:
#             lf = ''
#         qid2lf[qid] = lf
#
#     return qid2lf
#
#
# def print_qtexts(qids, qid2qtext):
#     for i, qid in enumerate(qids):
#         print("{}: {}".format(i, qid2qtext[qid]))
#
#
# def print_qtext_lf(qids, qid2qtext, qid2lf):
#     for i, qid in enumerate(qids):
#         print("{}: {} -- {}".format(i, qid2qtext[qid], qid2lf[qid]))
#
#
# def print_qtext_bothlfs(qids, qid2qtext, qid2lf1, qid2lf2):
#     for i, qid in enumerate(qids):
#         print("qid: {}".format(qid))
#         print("{}: {}\n{}\n{}\n".format(i, qid2qtext[qid], qid2lf1[qid], qid2lf2[qid]))
#
#
# def diff_in_lfs(qids, qid2lf1, qid2lf2):
#     num_diff_lf = 0
#     for i, qid in enumerate(qids):
#         lf1 = qid2lf1[qid]
#         lf2 = qid2lf2[qid]
#
#         if lf1 != lf2:
#             num_diff_lf += 1
#
#     return num_diff_lf
#
#
# def overlap_in_correct_incorrect_questext(correct_qids, incorrect_qids, qid2qtext):
#     correct_qtexts = set([qid2qtext[qid] for qid in correct_qids])
#     incorrect_qtexts = set([qid2qtext[qid] for qid in incorrect_qids])
#
#     overlap = correct_qtexts.intersection(incorrect_qtexts)
#
#     print("Unique correct questions: {}".format(len(correct_qtexts)))
#     print("Unique incorrect questions: {}".format(len(incorrect_qtexts)))
#     print("Unique question overlap: {}".format(len(overlap)))
#
#
# def error_overlap_statistics(file1, file2):
#     """Perform analysis on predictions of two different models and see the level of overlap in their predictions.
#
#
#     The two files are tab-separated, with the form
#         Question-id    Question-text    'C' or 'NC'     LogicalForm
#     The logical form is only for predictions from our models
#     """
#     instances1 = read_prediction_file(file1)
#     instances2 = read_prediction_file(file2)
#
#     qid2qtext1 = make_qid2questiondict(instances1)
#     qid2lf1 = make_qid2logicalform(instances1)
#
#     qid2qtext2 = make_qid2questiondict(instances2)
#     qid2lf2 = make_qid2logicalform(instances2)
#
#     num_instances1 = len(instances1)
#     num_instances2 = len(instances2)
#
#     correct_qids1, incorrect_qids1 = make_correct_incorrect_dicts(instances1)
#     correct_qids2, incorrect_qids2 = make_correct_incorrect_dicts(instances2)
#
#     perf1 = len(correct_qids1)/float(num_instances1)
#     perf2 = len(correct_qids2) / float(num_instances2)
#
#     correct_overlap_qids = correct_qids1.intersection(correct_qids2)
#
#     correct1_incorrect2_qids = correct_qids1.intersection(incorrect_qids2)
#     incorrect1_correct2_qids = incorrect_qids1.intersection(correct_qids2)
#
#     # For questions predicted correct by M1, and incorrect by M2 -- the number of ques with diff LFs
#     c1_nc2_lf_diff = diff_in_lfs(correct1_incorrect2_qids, qid2lf1, qid2lf2)
#     nc1_c2_lf_diff = diff_in_lfs(incorrect1_correct2_qids, qid2lf1, qid2lf2)
#
#     # print("Correct in Model 1, Incorrect in Model 2")
#     # print_qtext_bothlfs(correct1_incorrect2_qids, qid2qtext2, qid2lf1, qid2lf2)
#
#     print("Incorrect in Model 1, Correct in Model 2")
#     print_qtext_bothlfs(incorrect1_correct2_qids, qid2qtext1, qid2lf1, qid2lf2)
#
#
#     print("Model 1")
#     print("Num instances: {} Correct: {} Incorrect: {} Perf: {}".format(num_instances1, len(correct_qids1),
#                                                                         len(incorrect_qids1), perf1))
#     print()
#     print("Model 2")
#     print("Num instances: {} Correct: {} Incorrect: {} Perf: {}".format(num_instances2, len(correct_qids2),
#                                                                         len(incorrect_qids2), perf2))
#
#     print("Correct Overlap : {}".format(len(correct_overlap_qids)))
#
#     print("Correct in M1; Incorrect in M2: {}".format(len(correct1_incorrect2_qids)))
#     print("Num of ques w/ different LF predictions: {}".format(c1_nc2_lf_diff))
#     print()
#
#     print("Incorrect in M1; Correct in M2: {}".format(len(incorrect1_correct2_qids)))
#     print("Num of ques w/ different LF predictions: {}".format(nc1_c2_lf_diff))
#
#     print("Model1")
#     overlap_in_correct_incorrect_questext(correct_qids1, incorrect_qids1, qid2qtext1)
#
#     print("Model2")
#     overlap_in_correct_incorrect_questext(correct_qids2, incorrect_qids2, qid2qtext2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1')
    parser.add_argument('--file2')
    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2


    # OurGRUmodel = "/scratch1/nitishg/semqa/checkpoints/drop/date_num/date_numcq_hmvy_cnt_relprog_500/drop_parser/" \
    #               "TOKENS_qanet/ED_100/RG_1e-07/MODELTYPE_encoded/CNTFIX_false/aux_true/SUPEPOCHS_5/" \
    #               "S_10/CModelBM1" + "/predictions"

    OurBERTmodel = "./resources/semqa/checkpoints/drop/date_num/date_ydNEW_num_hmyw_cnt_rel_600/drop_parser_bert/" \
                   "CNTFIX_false/EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_5/S_100/BertModel_wTest/" \
                   "predictions/date_ydNEW_num_hmyw_cnt_rel_600_dev_numstepanalysis.tsv"

    NABERT_model = "./resources/semqa/checkpoints/drop-bert/mydata_ydNEW_rel/S_1000/BertModel_wTest/" \
                   "predictions/date_ydNEW_num_hmyw_cnt_rel_600_dev_pred.txt"

    # NABERTmodel = ("/scratch1/nitishg/semqa/checkpoints/drop-bert/mydata_ydre_relre/S_1/predictions")


    qid2steps = read_numsteps_annotation()

    qid2pred_outbert = read_qid2prediction(OurBERTmodel)

    qid2pred_nabert = read_qid2prediction(NABERT_model)


    print("\nOUR BERT")
    count_step2correct(qid2steps=qid2steps, qid2pred=qid2pred_outbert)

    print("\nNABERT")
    count_step2correct(qid2steps=qid2steps, qid2pred=qid2pred_nabert)
