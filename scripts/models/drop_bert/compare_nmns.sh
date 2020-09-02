#!/usr/bin/env

MODEL1=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v1-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10_S10/predictions/iclr_qdmr-v2_dev_predictions.jsonl
MODEL2=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v1-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42_FGS_S10/predictions/iclr_qdmr-v2_dev_predictions.jsonl
# ./resources/checkpoints/drop-iclr21/iclr_qdmr-v1-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_10/predictions/iclr_qdmr-v1_dev_predictions.jsonl

python -m analysis.compare_nmns --nmn_jsonl1 ${MODEL1} --nmn_jsonl2 ${MODEL2}