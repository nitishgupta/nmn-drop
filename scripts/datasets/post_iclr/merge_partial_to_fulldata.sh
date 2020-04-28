#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_post_iclr

FULLDATA_DIR=${ROOT_DIR}/preprocess

MYDATA_DIR=${ROOT_DIR}/date_num/date_yd_num_hmyw_cnt_whoarg

MERGED_DATA_DIR=${ROOT_DIR}/merged_data/date_yd_num_hmyw_cnt_whoarg_1200_full


python -m datasets.drop.fulldata_setting.merge_mydata  --mydata_dir ${MYDATA_DIR} \
                                                       --fulldata_dir ${FULLDATA_DIR} \
                                                       --merged_dir ${MERGED_DATA_DIR}
