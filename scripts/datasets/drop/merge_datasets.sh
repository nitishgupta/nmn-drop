#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_re

DIR1=date_num/dc_nc

DIR2=date/year_diff

OUTDIR=date_num/dc_nc_yeardiff


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}