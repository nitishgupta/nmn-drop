#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date/datecomp_full

DIR2=num/nc_howmanyyards_count_diff

OUTDIR=date_num/dc_nc_howmanyyards_count_diff


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}