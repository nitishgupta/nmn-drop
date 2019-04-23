#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DIR1=date/datecomp_pruned_augment_100

DIR2=num/years_after_passed

OUTDIR=date/datecomp100_yearsafterpassed


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}