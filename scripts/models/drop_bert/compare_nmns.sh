#!/usr/bin/env

MODEL1=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v2-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-FGS-DCYD-ND-v2-rev/predictions/iclr_qdmr-v2-noexc_test_predictions.jsonl
MODEL2=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v2-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_42/predictions/iclr_qdmr-v2-noexc_test_predictions.jsonl

python -m analysis.compare_nmns --nmn_jsonl1 ${MODEL1} --nmn_jsonl2 ${MODEL2}