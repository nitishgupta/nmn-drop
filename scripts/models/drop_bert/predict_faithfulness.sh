#!/usr/bin/env

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export INTERPRET=true

# SAVED MODEL
MODEL_DIR=./resources/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_42

PREDICTION_DIR=${MODEL_DIR}/predictions
MODEL_TAR=${MODEL_DIR}/model.tar.gz
mkdir ${PREDICTION_DIR}

DATASET_DIR=/shared/nitishg/data/drop/iclr21/
DATASET_NAME=faithful
FULL_VALFILE=${DATASET_DIR}/${DATASET_NAME}/iclr21_filter_faithful.json

FAITHFUL_PREDICTOR=drop_faithfulness_predictor
FAITHFULNESS_JSONL_PREDICTION_FILE=${PREDICTION_DIR}/iclr21_faithul_test_predictions.jsonl

METRICS_FILE=${PREDICTION_DIR}/iclr21_faithul_test_metrics.json

allennlp predict --output-file ${FAITHFULNESS_JSONL_PREDICTION_FILE} \
                 --predictor ${FAITHFUL_PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "interpret": true}, "validation_dataset_reader": { "mode": \"test\" }}" \
                 ${MODEL_TAR} ${FULL_VALFILE} &

allennlp evaluate --output-file ${METRICS_FILE} \
                  --cuda-device ${GPU} \
                  --include-package ${INCLUDE_PACKAGE} \
                  --overrides "{"model": {"beam_size": 1, "interpret": true}, "validation_dataset_reader": { "mode": \"test\" }}" \
                  ${MODEL_TAR} ${FULL_VALFILE}

printf "\n"
echo -e "Execution jsonl predictions file saved at: ${FAITHFULNESS_JSONL_PREDICTION_FILE}"
echo -e "Metrics file saved at: ${METRICS_FILE}"

python -m faithfulness_drop.compute_faithfulness_filter \
              --nmn_pred_jsonl ${PREDICTION_DIR}/iclr21_faithul_test_predictions.jsonl \
              --faithful_gold_json ${FULL_VALFILE}
