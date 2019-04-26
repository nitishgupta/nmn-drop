#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop_re

DIR1=num/numcomp_full

DIR2=num/longest_shortest_yards

OUTDIR=num/nc_longshortyards


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${ROOT_DIR}/${OUTDIR}