#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DIR1=date/datecomp_full

DIR2=date/year_diff_new

OUTDIR=date/datefull_yd_new


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}