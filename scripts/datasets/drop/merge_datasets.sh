#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=num/hmyw_filter

DIR2=num/who_relocate

OUTDIR=num/hmyw_who_relocate


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}