#!/usr/bin/env bash

PREPROCESS_DIR=/shared/nitishg/data/drop/preprocess

ICLR_SUBDATA=/shared/nitishg/data/drop/iclr21/iclr20_subsets

FULL_DATA_DIRNAME=iclr20_full-v1

ICLR21_DATA_DIR=/shared/nitishg/data/drop/iclr21

python -m datasets.drop.preprocess.datecomp.date_comparison_prune --input_dir ${PREPROCESS_DIR} \
                                                                  --output_dir ${ICLR_SUBDATA}/datecomp_prune


python -m datasets.drop.preprocess.datecomp.date_data_augmentation --input_dir ${ICLR_SUBDATA}/datecomp_prune \
                                                                   --output_dir ${ICLR_SUBDATA}/datecomp

# Remove temp datecomp-prune data
rm -r ${ICLR_SUBDATA}/datecomp_prune

# NUM-COMPARISON
python -m datasets.drop.preprocess.numcomp.make_numcomp  --input_dir ${PREPROCESS_DIR} \
                                                          --output_dir ${ICLR_SUBDATA}/numcomp

# YEAR-DIFF
python -m datasets.drop.preprocess.year_diff.year_diff  --input_dir ${PREPROCESS_DIR} \
                                                        --output_dir ${ICLR_SUBDATA}/year_diff

# WHO-ARG
python -m datasets.drop.preprocess.who_relocate.relocate_wprogs --input_dir ${PREPROCESS_DIR} \
                                                                --output_dir ${ICLR_SUBDATA}/who_arg

# HMYW (How Many Yards Was)
python -m datasets.drop.preprocess.how_many_yards.how_many_yards  --input_dir ${PREPROCESS_DIR} \
                                                                  --output_dir ${ICLR_SUBDATA}/hmyw

# COUNT
python -m datasets.drop.preprocess.how_many_yards.count_ques  --input_dir ${PREPROCESS_DIR} \
                                                              --output_dir ${ICLR_SUBDATA}/count


python -m datasets.drop.merge_datasets --dir1 ${ICLR_SUBDATA}/datecomp \
                                       --dir2 ${ICLR_SUBDATA}/year_diff \
                                       --outputdir ${ICLR_SUBDATA}/temp1


python -m datasets.drop.merge_datasets --dir1 ${ICLR_SUBDATA}/temp1 \
                                       --dir2 ${ICLR_SUBDATA}/numcomp \
                                       --outputdir ${ICLR_SUBDATA}/temp2


python -m datasets.drop.merge_datasets --dir1 ${ICLR_SUBDATA}/temp2 \
                                       --dir2 ${ICLR_SUBDATA}/hmyw \
                                       --outputdir ${ICLR_SUBDATA}/temp3


python -m datasets.drop.merge_datasets --dir1 ${ICLR_SUBDATA}/temp3 \
                                       --dir2 ${ICLR_SUBDATA}/count \
                                       --outputdir ${ICLR_SUBDATA}/temp4


python -m datasets.drop.merge_datasets --dir1 ${ICLR_SUBDATA}/temp4 \
                                       --dir2 ${ICLR_SUBDATA}/who_arg \
                                       --outputdir ${ICLR_SUBDATA}/temp5

mv ${ICLR_SUBDATA}/temp5 ${ICLR_SUBDATA}/drop_iclr_full_pre

rm -rf ${ICLR_SUBDATA}/temp1
rm -rf ${ICLR_SUBDATA}/temp2
rm -rf ${ICLR_SUBDATA}/temp3
rm -rf ${ICLR_SUBDATA}/temp4

python -m datasets.drop.preprocess.postprocess --input_dir ${ICLR_SUBDATA}/drop_iclr_full_pre \
                                               --output_dir ${ICLR_SUBDATA}/${FULL_DATA_DIRNAME}

rm -rf ${ICLR_SUBDATA}/drop_iclr_full_pre

# Copying final dataset to the outside directory
cp -r ${ICLR_SUBDATA}/${FULL_DATA_DIRNAME} ${ICLR21_DATA_DIR}
