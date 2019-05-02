#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date_num/dc_nc_yd_num

DIR2=num/synthetic_count_num

OUTDIR=date_num/dc_nc_yd_num_syn


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}