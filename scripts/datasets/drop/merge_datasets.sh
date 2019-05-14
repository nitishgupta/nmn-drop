#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date_num/date_numcq_hmvy_cnt_filter

DIR2=num/who_relocate

OUTDIR=date_num/date_numcq_hmvy_cnt_rel


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}