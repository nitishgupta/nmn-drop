#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=num/numcomp_100

DIR2=num/howmanyyards_count_diff

OUTDIR=num/nc100_howmanyyards_count_diff


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}