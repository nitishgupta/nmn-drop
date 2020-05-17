#!/usr/bin/env bash

ROOT_DIR=./resources/data//drop-w-qdmr

DATECOMP=datecomp_aug
YEAR_DIFF=year_diff
NUMCOMP=numcomp
HMYW=hmyw
COUNT=count
RELOCATE=who-arg

OUTPUT_DIR=${ROOT_DIR}/drop_iclr_full

rm -r ${ROOT_DIR}/temp1 ${ROOT_DIR}/temp2 ${ROOT_DIR}/temp3 ${ROOT_DIR}/temp4 ${ROOT_DIR}/temp5

python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATECOMP} \
                                       --dir2 ${ROOT_DIR}/${YEAR_DIFF} \
                                       --outputdir ${ROOT_DIR}/temp1


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/temp1 \
                                       --dir2 ${ROOT_DIR}/${NUMCOMP} \
                                       --outputdir ${ROOT_DIR}/temp2


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/temp2 \
                                       --dir2 ${ROOT_DIR}/${HMYW} \
                                       --outputdir ${ROOT_DIR}/temp3


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/temp3 \
                                       --dir2 ${ROOT_DIR}/${COUNT} \
                                       --outputdir ${ROOT_DIR}/temp4


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/temp4 \
                                       --dir2 ${ROOT_DIR}/${RELOCATE} \
                                       --outputdir ${ROOT_DIR}/temp5

mv ${ROOT_DIR}/temp5 ${OUTPUT_DIR}

rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp1
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp2
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp3
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp4


# Copying all individual datasets into the merged dataset's directory
mkdir ${OUTPUT_DIR}/questype_datasets

cp -r ${ROOT_DIR}/${DATECOMP}  ${OUTPUT_DIR}/questype_datasets
cp -r ${ROOT_DIR}/${YEAR_DIFF} ${OUTPUT_DIR}/questype_datasets
cp -r ${ROOT_DIR}/${NUMCOMP}   ${OUTPUT_DIR}/questype_datasets
cp -r ${ROOT_DIR}/${HMYW}      ${OUTPUT_DIR}/questype_datasets
cp -r ${ROOT_DIR}/${COUNT}     ${OUTPUT_DIR}/questype_datasets
cp -r ${ROOT_DIR}/${RELOCATE}  ${OUTPUT_DIR}/questype_datasets