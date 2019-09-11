#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DIR1=date_num/temp1

DIR2=num/how_many_yards_was

OUTDIR=date_num/temp2


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}