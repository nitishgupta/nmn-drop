#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DIR1=date_num/datecomp_numcomp_50

DIR2=num/year_diff

OUTDIR=date_num/dc_nc_50_yeardiff


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}