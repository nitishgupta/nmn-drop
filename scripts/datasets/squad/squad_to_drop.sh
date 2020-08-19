#!/usr/bin/env bash

#export OMP_NUM_THREADS=4
#export OPENBLAS_NUM_THREADS=4
#export OPENMP_NUM_THREADS=4
#export MKL_NUM_THREADS=4


python -m datasets.squad.squad_to_drop \
  --squad_json /shared/nitishg/data/squad/squad-train-v1.1.json \
  --squad_ques_jsonl /shared/nitishg/data/squad/squad-train-v1.1_questions.jsonl \
  --squad_ques_parse_jsonl /shared/nitishg/data/squad/squad-train-v1.1_questions_parse.jsonl \
  --output_json /shared/nitishg/data/squad/squad-train-v1.1_drop.json


python -m datasets.squad.squad_to_drop \
  --squad_json /shared/nitishg/data/squad/squad-dev-v1.1.json \
  --squad_ques_jsonl /shared/nitishg/data/squad/squad-dev-v1.1_questions.jsonl \
  --squad_ques_parse_jsonl /shared/nitishg/data/squad/squad-dev-v1.1_questions_parse.jsonl \
  --output_json /shared/nitishg/data/squad/squad-dev-v1.1_drop.json
