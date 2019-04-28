#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_s

DIR1=date/year_diff

DIR2=date_num/dc_nc

OUTDIR=date_num/dc_nc_yeardiff


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}