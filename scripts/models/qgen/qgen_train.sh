#!/usr/bin/env bash

CKPT_ROOT=/shared/nitishg/checkpoints/qgen

rm -rf ${CKPT_ROOT}


bash scripts/allennlp/train.sh training_config/qgen/qgen.jsonnet \
                               semqa \
                               ${CKPT_ROOT}