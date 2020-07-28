#!/usr/bin/env

NMN=/shared/nitishg/checkpoints/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_3_HEM_3_BM_1/S_10_newHP_qND/predictions/qdmr-v1_dev_predictions.jsonl
MTMSN=/shared/nitishg/checkpoints/MTMSN/drop-w-qdmr/qdmr-v1_iclrfull-ss/S_10/predictions/qdmr-v1_dev_preds.json

python -m analysis.compare_MTMSN_Ours --our_preds_jsonl ${NMN} --mtmsn_preds_json ${MTMSN}