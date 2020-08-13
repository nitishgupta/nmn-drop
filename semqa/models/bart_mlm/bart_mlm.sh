#!/usr/bin/env

INPUT_JSONL=semqa/qgen/bart_mlm/bart_input.jsonl
OUTPUT_JSONL=semqa/qgen/bart_mlm/bart_output.jsonl

allennlp predict \
  --output-file ${OUTPUT_JSONL} \
  --predictor "seq2seq" \
  --include-package semqa \
  --cuda-device 0 \
  /shared/nitishg/checkpoints/bart_mlm/model.tar.gz \
  ${INPUT_JSONL}
