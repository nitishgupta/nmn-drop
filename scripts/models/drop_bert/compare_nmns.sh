#!/usr/bin/env

MODEL1=./resources/checkpoints/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss-cnt/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_3_HEM_3_BM_1/S_10_PreBIO_SqNMN-Paired/predictions/qdmr-v2_dev_predictions.jsonl
MODEL2=./resources/checkpoints/drop-w-qdmr/qdmr-v2_iclrfull-v2-ss-cnt/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_3_HEM_3_BM_1/S_42_PreBIO_SqNMN/predictions/qdmr-v2_dev_predictions.jsonl

python -m analysis.compare_nmns --nmn_jsonl1 ${MODEL1} --nmn_jsonl2 ${MODEL2}