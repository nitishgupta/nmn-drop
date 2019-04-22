#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DIR1=date/datecomp_pruned_augment

DIR2=num/numcomp_prune_supervised

OUTDIR=date_num/datecomp_numcomp


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}