#!/usr/bin/env bash

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
