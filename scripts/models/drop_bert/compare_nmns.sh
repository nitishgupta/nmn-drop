#!/usr/bin/env

MODEL1=/shared/nitishg/checkpoints/drop-iclr21/diff_compsplit-v4/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_true/SUPEPOCHS_0_BM_1/S_42-FGS-DCYD-ND-MM/predictions/diff_compsplit-v4_test_predictions-Ex0-Rev_PP.jsonl
MODEL2=/shared/nitishg/checkpoints/drop-iclr21/diff_compsplit-v4/drop_parser_bert/Qattn_true/EXCLOSS_true/aux_false/IO_true/SHRDSUB_false/SUPEPOCHS_0_BM_1/S_10/predictions/diff_compsplit-v4_test_predictions-Ex0-Rev_PP.jsonl

python -m analysis.compare_nmns --nmn_jsonl1 ${MODEL1} --nmn_jsonl2 ${MODEL2}