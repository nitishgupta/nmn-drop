#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date/dc_yeardiff

DIR2=num/nc_hmyw_cnt_filter

OUTDIR=date_num/date_numcq_hmvy_cnt_filter


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}