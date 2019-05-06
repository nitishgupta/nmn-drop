#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date_num/datepaq_numcq_hmvy_ydiff

DIR2=num/yardscount_wqattn

OUTDIR=date_num/datepaq_numcq_hmvy_ydiff_countqa


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}