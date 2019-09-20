#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

PERC_SPLIT=0.2

FULLDATASET_DIR=${ROOT_DIR}/date_num/date_ydNEW_num_hmyw_cnt_rel_600
QTYPE_DATASETS_ROOTDIR=${ROOT_DIR}/date_num/date_ydNEW_num_hmyw_cnt_rel_600/questype_datasets


python -m datasets.drop.split_qtype_dev --fulldataset_dir=${FULLDATASET_DIR} \
                                        --root_qtype_datasets_dir=${QTYPE_DATASETS_ROOTDIR}
