#!/usr/bin/env bash

ROOT_DIR=./resources/data/drop

DATASETS_TO_MERGE=[]

DIR1=num/how_many_years/how_many_years_after_the

DIR2=num/how_many_years/how_many_years_did_it

REST_DIRS=(num/how_many_years/how_many_years_did_the num/how_many_years/how_many_years_passed_between \
           num/how_many_years/how_many_years_was)

OUTDIR=num/year_diff

TMP_DIR=$(mktemp -u -d)
echo ${TMP_DIR}

for val1 in ${REST_DIRS[*]}; do
     echo $val1
done


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIR1} \
                                       --dir2 ${ROOT_DIR}/${DIR2} \
                                       --outputdir ${TMP_DIR}


for DIRNAME in ${REST_DIRS[*]};
do
    NEW_TMP_DIR=$(mktemp -u -d)
    echo ${NEW_TMP_DIR}
    python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DIRNAME} \
                                       --dir2 ${TMP_DIR} \
                                       --outputdir ${NEW_TMP_DIR}
    TMP_DIR=${NEW_TMP_DIR}
    echo ${TMP_DIR}
done


if [ -d ${ROOT_DIR}/${OUTDIR} ]; then
    echo ${ROOT_DIR}/${OUTDIR}
    echo "NO merging happened"
    echo "OUTPUT DIR EXISTS: ${ROOT_DIR}/${OUTDIR}"
    echo "Check if empty; delete; and re-run the code"
else
    echo "Making ${ROOT_DIR}/${OUTDIR} and copying merged data"
    mkdir ${ROOT_DIR}/${OUTDIR}
    mv ${TMP_DIR}/* ${ROOT_DIR}/${OUTDIR}/
fi





