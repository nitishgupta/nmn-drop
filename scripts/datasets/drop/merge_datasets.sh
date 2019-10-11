#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_acl

DIR1=date_num/date_yd_num_hmyw_cnt_whoarg_600

DIR2=num/yardsdiff

OUTDIR=date_num/iclr20_yardsdiff


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}