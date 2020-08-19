#!/usr/bin/env bash

export TMPDIR=/shared/nitishg/tmp
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OPENMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

TRAIN_JSONL=/shared/nitishg/data/squad/squad-train-v1.1_questions.jsonl
OUTPUT_TRAIN_JSONL=/shared/nitishg/data/squad/squad-train-v1.1_questions_depparse.jsonl

DEV_JSONL=/shared/nitishg/data/squad/squad-dev-v1.1_questions.jsonl
OUTPUT_DEV_JSONL=/shared/nitishg/data/squad/squad-dev-v1.1_questions_depparse.jsonl

allennlp predict \
  --output-file ${OUTPUT_TRAIN_JSONL} \
  --batch-size=64 \
  --cuda-device 0 \
  --predictor my_biaffine_dependency_parser \
  --include-package semqa \
  --silent \
  "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz" \
  ${TRAIN_JSONL}


allennlp predict \
  --output-file ${OUTPUT_DEV_JSONL} \
  --batch-size=64 \
  --cuda-device 0 \
  --predictor my_biaffine_dependency_parser \
  --include-package semqa \
  --silent \
  "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz" \
  ${DEV_JSONL}




