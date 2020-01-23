#!/usr/bin/env bash

# THIS SHOULD MAKE THE DATASET FOR ICLR20 submission

ROOT_DIR=/shared/nitishg/data/drop_post_iclr

PREPROCESS_DIR=preprocess

PRUNE_DATECOMP=date/datecomp_prune
DATECOMP=date/datecomp_full
PRUNE_NUMCOMP=num/numcomp_prune
NUMCOMP=num/numcomp_full
YEAR_DIFF=date/year_diff
WHO_ARG=num/who_arg_nov19
HMYW=num/how_many_yards_was_nov19
COUNT=num/count_nov19

DATE_NUM_DIR=date_num

DATASET_FULL_ANNO=date_num/date_yd_num_hmyw_cnt_whoarg_nov19

ANNOTATION_FOR_PARAS=800
DATASET_PRUNED_ANNO=date_num/date_yd_num_hmyw_cnt_whoarg_nov19_${ANNOTATION_FOR_PARAS}

# Into my dev and mytest
DEV_TEST_SPLIT_RATIO=0.2


# DATE-COMPARISON
python -m datasets.drop.preprocess.datecomp.date_comparison_prune --input_dir ${ROOT_DIR}/${PREPROCESS_DIR} \
                                                                  --output_dir ${ROOT_DIR}/${PRUNE_DATECOMP}


python -m datasets.drop.preprocess.datecomp.date_data_augmentation --input_dir ${ROOT_DIR}/${PRUNE_DATECOMP} \
                                                                   --output_dir ${ROOT_DIR}/${DATECOMP}

# Remove temp datecomp-prune data
rm -r ${ROOT_DIR:?}/${PRUNE_DATECOMP:?}

# NUM-COMPARISON
python -m datasets.drop.preprocess.numcomp.prune_numcomp  --input_dir ${ROOT_DIR}/${PREPROCESS_DIR} \
                                                          --output_dir ${ROOT_DIR}/${PRUNE_NUMCOMP}


python -m datasets.drop.preprocess.numcomp.add_supervision  --input_dir ${ROOT_DIR}/${PRUNE_NUMCOMP} \
                                                            --output_dir ${ROOT_DIR}/${NUMCOMP}

# Remove temp numcomp-prune data
rm -r ${ROOT_DIR:?}/${PRUNE_NUMCOMP:?}

# YEAR-DIFF
python -m datasets.drop.preprocess.year_diff.year_diff  --input_dir ${ROOT_DIR}/${PREPROCESS_DIR} \
                                                        --output_dir ${ROOT_DIR}/${YEAR_DIFF}

# WHO-ARG
python -m datasets.drop.preprocess.who_relocate.relocate_wprogs_nov19 --input_dir ${ROOT_DIR}/${PREPROCESS_DIR} \
                                                                      --output_dir ${ROOT_DIR}/${WHO_ARG}

# HMYW (How Many Yards Was)
python -m datasets.drop.preprocess.how_many_yards.how_many_yards_nov19  --input_dir ${ROOT_DIR}/${PREPROCESS_DIR} \
                                                                        --output_dir ${ROOT_DIR}/${HMYW} \
                                                                        --qattn \
                                                                        --numground

# COUNT
python -m datasets.drop.preprocess.how_many_yards.count_ques_nov19    --input_dir ${ROOT_DIR}/${PREPROCESS_DIR} \
                                                                      --output_dir ${ROOT_DIR}/${COUNT}




python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATECOMP} \
                                       --dir2 ${ROOT_DIR}/${YEAR_DIFF} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp1


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp1 \
                                       --dir2 ${ROOT_DIR}/${NUMCOMP} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp2


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp2 \
                                       --dir2 ${ROOT_DIR}/${HMYW} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp3


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp3 \
                                       --dir2 ${ROOT_DIR}/${COUNT} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp4


python -m datasets.drop.merge_datasets --dir1 ${ROOT_DIR}/${DATE_NUM_DIR}/temp4 \
                                       --dir2 ${ROOT_DIR}/${WHO_ARG} \
                                       --outputdir ${ROOT_DIR}/${DATE_NUM_DIR}/temp5

mv ${ROOT_DIR}/${DATE_NUM_DIR}/temp5 ${ROOT_DIR}/${DATASET_FULL_ANNO}


rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp1
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp2
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp3
rm -r ${ROOT_DIR}/${DATE_NUM_DIR}/temp4


# Copying all individual datasets into the merged dataset's directory
mkdir ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets


cp -r ${ROOT_DIR}/${DATECOMP}  ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets
cp -r ${ROOT_DIR}/${YEAR_DIFF} ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets
cp -r ${ROOT_DIR}/${NUMCOMP}   ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets
cp -r ${ROOT_DIR}/${HMYW}      ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets
cp -r ${ROOT_DIR}/${COUNT}     ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets
cp -r ${ROOT_DIR}/${WHO_ARG}   ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets


python -m datasets.drop.remove_strong_supervision --input_dir ${ROOT_DIR}/${DATASET_FULL_ANNO} \
                                                  --output_dir ${ROOT_DIR}/${DATASET_PRUNED_ANNO} \
                                                  --annotation_for_numpassages ${ANNOTATION_FOR_PARAS}
#
#
cp -r ${ROOT_DIR}/${DATASET_FULL_ANNO}/questype_datasets ${ROOT_DIR}/${DATASET_PRUNED_ANNO}/questype_datasets


python -m datasets.drop.split_dev_ratio --fulldataset_dir=${ROOT_DIR}/${DATASET_PRUNED_ANNO} \
                                        --qtype_dir_name=questype_datasets \
                                        --split_ratio=${DEV_TEST_SPLIT_RATIO}