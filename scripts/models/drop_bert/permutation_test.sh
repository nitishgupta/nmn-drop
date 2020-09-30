#!/usr/bin/env

# Baseline MODEL
NMN_JSONL_1=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_42/predictions/iclr_qdmr-v4-noexc_test_predictions-Ex0-Rev.jsonl
# Our MODEL
NMN_JSONL_2=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-CONS-MODGEN-FOUND-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl


# CONS
# /shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-FGS-DCYD-ND-MM/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

# FOUND
# /shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-FOUND-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

# MODGEN
# /shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-MODGEN-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

# CONS-FOUND
# /shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-FOUND-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

# FOUND-MODGEN
# /shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-FOUND-MODGEN-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

# CONS-MODGEN
# /shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-MODGEN-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

# CONS-MODGEN-FOUND
# /shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-CONS-MODGEN-FOUND-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl


python -m analysis.paired_permutation_test ${NMN_JSONL_1}  ${NMN_JSONL_2}