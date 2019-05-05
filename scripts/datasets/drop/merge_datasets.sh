#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=num/nc_hmyw

DIR2=date/datecomp_full

OUTDIR=date_num/dc_nc_hmyw


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}