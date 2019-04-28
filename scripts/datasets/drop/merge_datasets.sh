#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date/datecomp_100

DIR2=date/year_diff

OUTDIR=date/dc_100_yeardiff


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}