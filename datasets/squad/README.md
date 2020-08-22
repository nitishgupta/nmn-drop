# Squad preprocessing

### DEPRECATED `Question constituency parsing`
To run a pre-trained constituency parser;

#### `Convert questions to JSONL format`
```
python -m datasets.squad.question_to_jsonl \
    --squad_train_json /shared/nitishg/data/squad/squad-train-v1.1.json
    --squad_dev_json /shared/nitishg/data/squad/squad-dev-v1.1.json
```  

#### `Constituency parsing`
```
bash scripts/datasets/squad/question_constituency_parse.sh
```
This writes `/shared/nitishg/data/squad/squad-train-v1.1_questions_parse.jsonl` and `dev`
with question's parse in `JSONL` format.

## Convert SQuAD to DROP format

Input: original squad `json` dataset

Output: SquAD dataset in DROP `json` format with gold-program-supervision, 
`project(select)` for all questions w/o question-attention
```
python -m datasets.squad.squad_to_drop --squad_datadir /shared/nitishg/data/squad
```

Outputs `/shared/nitishg/data/squad/squad-train-v1.1_drop.json` and the dev dataset.

## Paired Data

### Train BART-based Question-Generation model
Input: SQuAD dataset in DROP format

Output: Fine-tuned BART that generates question given a passage marked 
w/ an answer span

```
bash scripts/models/squad_qgen.sh
``` 

Output model: `/shared/nitishg/checkpoints/squad-qgen/BS_6/BEAM_1/MASKQ_false/S_42/model.tar.gz`


### Generate paired SQuAD data
Use `datasets/squad/qgen/generate_contrastive_question.py` w/ squad in DROP format and BART-QGen model to generate
`squad-train-v1.1_drop-wcontrastive.json`. 