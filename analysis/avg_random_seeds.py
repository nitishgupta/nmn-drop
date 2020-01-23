import os
import glob
import json
from collections import defaultdict

"""This path should have multiple folders for different seeds, and within each seed a directory for ModelName. 
For eg. SERIALIZATION_PATH/S_1/BertModel/, SERIALIZATION_PATH/S_2/BertModel/, etc. 

From inside each model folder this script expects a predictions folder. 
"""
# SERIALIZATION_PATH="./resources/semqa/checkpoints/drop-bert/mydata_ydNEW_rel"
SERIALIZATION_PATH = "./resources/semqa/checkpoints/drop/date_num/date_yd_num_hmyw_cnt_whoarg_600/drop_parser_bert/" \
                    "EXCLOSS_true/MMLLOSS_true/aux_true/SUPEPOCHS_5"
glob_path = SERIALIZATION_PATH + "/S_*/" + "BeamSize1"

print("\nSERIALIZATION PATH: {}".format(SERIALIZATION_PATH))

# glob_path = SERIALIZATION_PATH

paths = glob.glob(glob_path)
seeds = []

print(paths)

seedpath2valf1 = {}
seedpath2valem = {}
for seedpath in paths:
    metrics_glob_path = os.path.join(seedpath, "metrics_epoch_??.json")
    metrics_paths = glob.glob(metrics_glob_path)
    if not metrics_paths:
        continue
    latest_metric_path = sorted(metrics_paths, reverse=True)[0]
    metric_dict = json.load(open(latest_metric_path))
    best_f1 = metric_dict["best_validation_f1"]
    best_em = metric_dict["best_validation_em"]
    seedpath2valf1[seedpath] = best_f1
    seedpath2valem[seedpath] = best_em


num_seeds = len(seedpath2valf1)
print("Total number of seeds: {}".format(num_seeds))

if num_seeds > 0:
    print("Latest Metric Val F1")
    for k, v in seedpath2valf1.items():
        print("{} : {}".format(k, v))

    print("Latest Metric Test EM")
    for k, v in seedpath2valem.items():
        print("{} : {}".format(k, v))

    avg_val_f1 = sum(x[1] for x in seedpath2valf1.items()) / len(seedpath2valf1)
    avg_val_em = sum(x[1] for x in seedpath2valem.items()) / len(seedpath2valem)

    print("Avg F1: {}".format(avg_val_f1))
    print("Avg EM: {}".format(avg_val_em))
    print()

seedpath2datasetf1 = {}
seedpath2datasetem = {}

for seedpath in paths:
    predictions_dir = os.path.join(seedpath, "predictions")
    if not os.path.exists(predictions_dir):
        print("Predictions dir does not exist!\n{}".format(predictions_dir))
    globpath = os.path.join(predictions_dir, "*_metrics.json")
    dataset_eval_filepaths = glob.glob(globpath)
    datasetname2f1 = {}
    datasetname2em = {}
    for filepath in dataset_eval_filepaths:
        filename = os.path.split(filepath)[1]
        eval_dict = json.load(open(filepath, "r"))
        datasetname2f1[filename] = eval_dict["f1"]
        datasetname2em[filename] = eval_dict["em"]

    seedpath2datasetf1[seedpath] = datasetname2f1
    seedpath2datasetem[seedpath] = datasetname2em


num_seeds = len(seedpath2datasetf1)

avg_dataset_f1_perf = defaultdict(float)
avg_dataset_em_perf = defaultdict(float)

for seeddir, dataset2f1 in seedpath2datasetf1.items():
    for evaldataset, f1 in dataset2f1.items():
        avg_dataset_f1_perf[evaldataset] += f1
for dataset, f1 in avg_dataset_f1_perf.items():
    avg_dataset_f1_perf[dataset] = f1 / num_seeds


for seeddir, dataset2em in seedpath2datasetem.items():
    for evaldataset, em in dataset2em.items():
        avg_dataset_em_perf[evaldataset] += em
for dataset, em in avg_dataset_em_perf.items():
    avg_dataset_em_perf[dataset] = em / num_seeds

print("Avg F1")
for k, v in avg_dataset_f1_perf.items():
    print("{} : {}".format(k, v))

print("Avg EM")
for k, v in avg_dataset_em_perf.items():
    print("{} : {}".format(k, v))
