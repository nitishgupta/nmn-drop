#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DATECOMP=date/datecomp_full
YEAR_DIFF=date/year_diff_new2
NUMCOMP=num/numcomp_full
HMYW=num/how_many_yards_was
COUNT=num/count
RELOCATE=num/who_relocate


DATE_NUM_DIR=${ROOT_DIR}/date_num

OUTPUT_DIR=${DATE_NUM_DIR}/date_ydnew2_num_hmyw_cnt_rel

python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATECOMP} \
                                       --dir2 ${ROOT_DIR}/${YEAR_DIFF} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp1


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp1 \
                                       --dir2 ${ROOT_DIR}/${NUMCOMP} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp2


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp2 \
                                       --dir2 ${ROOT_DIR}/${HMYW} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp3


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp3 \
                                       --dir2 ${ROOT_DIR}/${COUNT} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp4


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp4 \
                                       --dir2 ${ROOT_DIR}/${RELOCATE} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp5

mv ${ROOT_DIR}/${DATE_NUM_DIR}/temp5 ${OUTPUT_DIR}

rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp1
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp2
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp3
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp4



