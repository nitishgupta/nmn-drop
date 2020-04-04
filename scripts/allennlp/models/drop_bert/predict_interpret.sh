#!/usr/bin/env

# PACKAGE TO BE INCLUDED WHICH HOUSES ALL THE CODE
INCLUDE_PACKAGE=semqa
export GPU=0
export BEAMSIZE=1
export DEBUG=true
export INTERPRET=true

# SAVED MODEL
MODEL_DIR=/shared/nitishg/semqa/checkpoints/drop/merged_data/date_yd_num_hmyw_cnt_whoarg_1200_full/drop_parser_bert/EXCLOSS_false/MMLLOSS_true/aux_true/SUPEPOCHS_3/S_10/BeamSize2
MODEL_TAR=${MODEL_DIR}/model.tar.gz
PREDICTION_DIR=${MODEL_DIR}/predictions
mkdir ${PREDICTION_DIR}

# ICLR DEV
#DEV_FILE=./resources/data/drop_post_iclr/date_num/nov19_1500/drop_dataset_dev.json
#VAL_METRICS_FILE=${PREDICTION_DIR}/iclr_dev_metrics.json

## FULL DEV
#DEV_FILE=./resources/data/drop_post_iclr/preprocess/drop_dataset_dev.json
#VAL_METRICS_FILE=${PREDICTION_DIR}/drop_dev_metrics.json

#allennlp evaluate --output-file ${VAL_METRICS_FILE} \
#                  --cuda-device ${GPU} \
#                  --include-package ${INCLUDE_PACKAGE} \
#                  ${MODEL_TAR} ${DEV_FILE}


INTERPRET_DEV_FILE=/shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno.json
#
## Prediction output including verbose execution logs
PREDICTOR=drop_interpret_predictor
PREDICTION_FILE=${PREDICTION_DIR}/interpret_dev_moduleoutputs.jsonl
#
allennlp predict --output-file ${PREDICTION_FILE} \
                 --predictor ${PREDICTOR} \
                 --cuda-device ${GPU} \
                 --include-package ${INCLUDE_PACKAGE} \
                 --silent \
                 --batch-size 4 \
                 --use-dataset-reader \
                 --overrides "{"model": { "beam_size": ${BEAMSIZE}, "debug": ${DEBUG}, "interpret": ${INTERPRET}}}" \
                 ${MODEL_TAR} ${INTERPRET_DEV_FILE}
#
#echo -e "Dev predictions file saved at: ${PREDICTION_FILE}"
#
python -m interpret_drop.compute_interpretability \
  --module_output_anno_json /shared/nitishg/data/interpret-drop/interpret_dev/interpret_dev_manual_wanno.json \
  --module_output_pred_jsonl ${PREDICTION_FILE}
