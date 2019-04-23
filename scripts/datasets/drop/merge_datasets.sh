#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DIR1=num/numcomp_prune_supervised_100

DIR2=date/datecomp100_yearsafterpassed

OUTDIR=date_num/dc_nc_100_yearspassedafter


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}