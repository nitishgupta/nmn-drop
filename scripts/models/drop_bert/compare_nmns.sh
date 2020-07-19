#!/usr/bin/env

MODEL1=/shared/nitishg/checkpoints/drop-w-qdmr/iclr-full-qdmr-filter-v2-ss/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_true/IO_true/SHRDSUB_true/SUPEPOCHS_5_HEM_5_BM_1/S_1337/predictions/qdmr-filter-v2-ss_dev_predictions.jsonl
MODEL2=/shared/nitishg/checkpoints/drop-w-qdmr/iclr-full-qdmr-filter-v2-ss/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_true/IO_true/SHRDSUB_false/SUPEPOCHS_5_HEM_5_BM_1/S_1337/predictions/qdmr-filter-v2-ss_dev_predictions.jsonl

python -m analysis.compare_nmns --nmn_jsonl1 ${MODEL1} --nmn_jsonl2 ${MODEL2}