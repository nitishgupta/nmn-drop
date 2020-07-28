#!/usr/bin/env

MODEL1=/shared/nitishg/checkpoints/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_3_HEM_3_BM_1/S_10_newHP_qND/predictions/qdmr-v1_dev_predictions.jsonl
MODEL2=/shared/nitishg/checkpoints/drop-w-qdmr/qdmr-v1_iclrfull-ss/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_3_HEM_3_BM_1/S_10_newHP_qND_RN/predictions/qdmr-v1_dev_predictions.jsonl

python -m analysis.compare_nmns --nmn_jsonl1 ${MODEL1} --nmn_jsonl2 ${MODEL2}