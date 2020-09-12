#!/usr/bin/env

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true
export INTERPRET=true

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/drop-iclr21/diff_compsplit-v3/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-FGS-DCYD-ND-MM

PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop/iclr21
DATASET_NAME=diff_compsplit-v3
# minlt_zeroqsplit-v2
# iclr_qdmr-v2-noexc

# train/dev/test
SPLIT=test
# paired
# dev

FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/drop_dataset_${SPLIT}.json
# paired_questions.json

VIS_PREDICTOR=drop_parser_predictor
VIS_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${SPLIT}_visualize_GP.txt

JSONL_PREDICTOR=drop_parser_jsonl_predictor
JSONL_PREDICTION_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${SPLIT}_predictions_GP.jsonl

METRICS_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${SPLIT}_metrics_GP.json

EVAL_TXT_FILE=${PREDICTION_DIR}/${DATASET_NAME}_${SPLIT}_eval_GP.txt

allennlp predict --output-file ${VIS_PREDICTION_FILE} \
                 --predictor ${VIS_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "interpret": ${INTERPRET}}, "validation_dataset_reader": { "mode": \"test\" }}" \
                 ${MODEL_TAR} ${FULL_VALFILE} &

allennlp predict --output-file ${JSONL_PREDICTION_FILE} \
                 --predictor ${JSONL_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}, "interpret": ${INTERPRET}}}" \
                 ${MODEL_TAR} ${FULL_VALFILE} &

allennlp evaluate --output-file ${METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}, "interpret": ${INTERPRET}}}" \
                  ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Vis predictions file saved at: ${VIS_PREDICTION_FILE}"
echo -e "Jsonl predictions file saved at: ${JSONL_PREDICTION_FILE}"
echo -e "Metrics file saved at: ${METRICS_FILE}"

python -m analysis.qtype_eval --nmn_jsonl ${JSONL_PREDICTION_FILE} --output_file ${EVAL_TXT_FILE}
