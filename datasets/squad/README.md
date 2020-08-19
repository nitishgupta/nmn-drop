# Squad preprocessing

### Question constituency parsing
To run a pre-trained constituency parser;

#### Convert questions to JSONL format
```
python -m datasets.squad.question_to_jsonl \
    --squad_train_json /shared/nitishg/data/squad/squad-train-v1.1.json
    --squad_dev_json /shared/nitishg/data/squad/squad-dev-v1.1.json
```  

#### Constituency parsing
```
bash scripts/datasets/squad/question_constituency_parse.sh
```
This writes `/shared/nitishg/data/squad/squad-train-v1.1_questions_parse.jsonl` and `dev`
with question's parse in `JSONL` format.

### Convert SQuAD to DROP format

Input: original squad `json` dataset, questions and their parses in `jsonl`

Output: SquAD dataset in DROP `json` format with gold-program-supervision 
based on WH-phrase from the constituency parse.
```
bash scripts/datasets/squad/squad_to_drop.sh
```

Outputs `/shared/nitishg/data/squad/squad-train-v1.1_drop.json` and the dev dataset.
