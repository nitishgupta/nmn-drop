from typing import List
import sys
import numpy as np
from semqa.utils.prediction_analysis import NMNPredictionInstance, read_nmn_prediction_file, avg_f1, get_correct_qids, \
    avg_em

nmn1_jsonl = sys.argv[1]
nmn2_jsonl = sys.argv[2]

print("Reading NMN predictions ...")
nmn1_preds: List[NMNPredictionInstance] = read_nmn_prediction_file(nmn1_jsonl)
nmn2_preds: List[NMNPredictionInstance] = read_nmn_prediction_file(nmn2_jsonl)

assert len(nmn1_preds) == len(nmn2_preds)

qids = list([instance.query_id for instance in nmn1_preds])

qid2nmn1 = {instance.query_id: instance for instance in nmn1_preds}
qid2nmn2 = {instance.query_id: instance for instance in nmn2_preds}

# To keep instance-F1s in the same order for the two models
nmn1_f1s = [qid2nmn1[qid].f1_score for qid in qids]
nmn2_f1s = [qid2nmn2[qid].f1_score for qid in qids]

print("NMN 1: {}".format(np.mean(nmn1_f1s)))
print("NMN 2: {}".format(np.mean(nmn2_f1s)))

both_scores = [nmn2_f1s, nmn1_f1s]

original_statistic = np.mean(both_scores[0])-np.mean(both_scores[1])
print(f"Original mean (statistic): {original_statistic}")


sample_size = 10000
print("Running permutation test w/ trial_size: {}".format(sample_size))
signs = np.random.binomial(1, 0.5, size=(sample_size, len(nmn1_f1s)))
num_exceeding = 0
for trial in range(signs.shape[0]):
    if trial % 1000 == 0:
        print("trials run: {}".format(trial))

    precisions = [[], []]
    for j in range(len(both_scores[0])):
        if signs[trial][j] > 0.5:
            precisions[0].append(nmn2_f1s[j])
            precisions[1].append(nmn1_f1s[j])
        else:
            precisions[0].append(nmn1_f1s[j])
            precisions[1].append(nmn2_f1s[j])
    # for i in range(2):
    #     for j in range(len(both_scores[i])):
    #         if signs[trial][j] > 0.5:
    #             precisions[i].append(nmn1_f1s[j])
    #         else:
    #             precisions[1-i].append(nmn2_f1s[j])
    assert len(precisions[0]) == len(precisions[1])
    statistic = np.mean(precisions[0])-np.mean(precisions[1])
    # if abs(statistic) >= original_statistic:
    if statistic >= original_statistic:
        num_exceeding += 1

print('p-value: '+str(float(num_exceeding)/signs.shape[0]))