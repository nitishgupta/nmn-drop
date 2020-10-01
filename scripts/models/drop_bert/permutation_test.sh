#!/usr/bin/env


BASELINE=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_42/predictions/iclr_qdmr-v4-noexc_test_predictions-Ex0-Rev.jsonl

CONS=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-FGS-DCYD-ND-MM/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

FOUND=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-FOUND-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

MODGEN=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-MODGEN-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

CONSFOUND=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-FOUND-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

FOUNDMODGEN=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-FOUND-MODGEN-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

CONSMODGEN=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-MODGEN-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

CONSMODGENFOUND=/shared/nitishg/checkpoints/drop-iclr21/iclr_qdmr-v4-noexc/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-CONS-MODGEN-FOUND-06/predictions/iclr_qdmr-v4-noexc_test_predictions.jsonl

MTMSN=/shared/nitishg/checkpoints/MTMSN/iclr21/iclr_qdmr-v4-noexc/S_10/predictions/iclr_qdmr-v4-noexc_test_preds.json


# DIFF - COMP-GEN
MTMSN_DIFFCOMP=/shared/nitishg/checkpoints/MTMSN/iclr21/diff_compsplit-v4/S_42/predictions/diff_compsplit-v4_test_preds.json
CONSMODGENFOUND_DIFFCOMP=/shared/nitishg/checkpoints/drop-iclr21/diff_compsplit-v4/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-MODGEN-FOUND-06/predictions/diff_compsplit-v4_test_predictions.jsonl
CONSMODGENFOUND_DIFFCOMP_GP=/shared/nitishg/checkpoints/drop-iclr21/diff_compsplit-v4/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-MODGEN-FOUND-06/predictions/diff_compsplit-v4_test_predictions_GP.jsonl

# FILTER COMP-GEN
MTMSN_FILTER=/shared/nitishg/checkpoints/MTMSN/iclr21/filter_compsplit-v4/S_10/predictions/filter_compsplit-v4_test_preds.json
CONSMODGENFOUND_FILTER=/shared/nitishg/checkpoints/drop-iclr21/filter_compsplit-v4/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-MODGEN-FOUND-06/predictions/filter_compsplit-v4_test_predictions.jsonl
CONSMODGENFOUND_FILTER_GP=/shared/nitishg/checkpoints/drop-iclr21/filter_compsplit-v4/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_10-CONS-MODGEN-FOUND-06/predictions/filter_compsplit-v4_test_predictions_GP.jsonl

NMN_JSONL_1=${BASELINE}
NMN_JSONL_2=${CONSMODGENFOUND}

python -m analysis.paired_permutation_test ${NMN_JSONL_1}  ${NMN_JSONL_2}