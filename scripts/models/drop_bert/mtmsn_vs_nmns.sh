#!/usr/bin/env

NMN=/shared/nitishg/checkpoints/drop-w-qdmr/iclr-full-qdmr-filter-v2-ss/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_true/IO_true/SHRDSUB_true/SUPEPOCHS_5_HEM_5_BM_1/S_1337/predictions/qdmr-filter-v2-ss_dev_predictions.jsonl
MTMSN=./resources/checkpoints/MTMSN/drop-w-qdmr/iclr-full-qdmr-filter-v2-ss/S_42/predictions/qdmr-filter-v2_dev_preds.json

python -m analysis.compare_MTMSN_Ours --our_preds_jsonl ${NMN} --mtmsn_preds_json ${MTMSN}