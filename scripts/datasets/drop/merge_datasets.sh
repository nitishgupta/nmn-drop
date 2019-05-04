#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date_num/dcnc100_yeardiff_hmvyqatnumgr_ydiff

DIR2=num/yardscount

OUTDIR=date_num/dcnc100_yeardiff_hmvyqatnumgr_ydiff_count


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}