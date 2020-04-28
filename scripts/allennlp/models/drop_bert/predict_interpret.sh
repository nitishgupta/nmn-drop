#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true
export INTERPRET=true

# SAVED MODEL
MODEL_DIR=/shared/nitishg/semqa/checkpoints/drop/generalization/min_max/drop_parser_bert/EXCLOSS_false/MMLLOSS_true/aux_true/SUPEPOCHS_0/S_1/BeamSize1_HEM0
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

INTERPRET_DEV_FILE=/shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno_v2.json

# Prediction output including verbose execution logs
PREDICTOR=drop_interpret_predictor
PREDICTION_FILE=${PREDICTION_DIR}/interpret_dev_moduleoutputs.jsonl

allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}, "interpret": ${INTERPRET}}}" \
                 ${MODEL_TAR} ${INTERPRET_DEV_FILE}


python -m interpret_drop.compute_interpretability \
  --module_output_anno_json ${INTERPRET_DEV_FILE} \
  --module_output_pred_jsonl ${PREDICTION_FILE}
