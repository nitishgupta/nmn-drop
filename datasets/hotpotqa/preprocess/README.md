# Preprocessing HotpotQA


## Commands to run:
1. `scripts/datasets/hotpotqa/raw_gold_contexts.sh` - 
Makes raw json files for train and dev_distractor that contain only gold contexts as given in the dataset.

2. `scripts/datasets/hotpotqa/preprocess.sh` -
Preprocess (as explained below) the different splits of the dataset.

3. `scripts/datasets/hotpotqa/extract_bool_ques.sh` -
Extracts boolean questions for different splits.

* * *

## Tokenize --- ```tokenize_mp```
Main function: ```processJsonObj``` --- runs in multiprocesses

1. Question: Tokenize and NER 
2. Context: For each sentence, Tokenize and NER

For both above, remove multiple white spaces, and also store whitespace info to recover the un-tokenized text back.

Following fields are added or modified:
1. ```constants.q_field```, ```q_whitespaces_field```, ```q_ner_field```
2. ```context_field```, ```context_whitespaces_field```, ```context_ner_field```
3. ```constants.ans_tokenized_field``` (used for grounding) and ```ans_ner_field``` (ans ner is not used)  



## Normalize and clean NE mentions --- ```clean_ners_mp```
Main function: ```processJsonObj```, ```cleanNERList```, ```normalizers``` for each NE type
--- runs in multiprocesses.

Normalize types of NE mentions and their values,
for both contexts and questions

Instead of one ners list (for both ques and contexts), now have three different lists:
Each for Entity, Num, and Date

1. Remove ```REMOVE_NER_TYPES``` types of mentions
2. Other mentions get resolved to ```constants.ENTITY```, ```constants.NUM```, or ```constants.DATE``` 
3. Get a ```NUMBER_DICT``` and ```DATE_DICT``` --- contains mappings from string to normalized value.

Apart from now having three different NER lists, following fields are added.
They contain dict from  string to normalized value
1. ```constants.nums_normalized_field```
2. ```constants.dates_normalized_field```


## Flatten Contexts --- ```flatten_contexts```
Main function: ```flattenContexts```

Merge context sentences into a single sentence.
Accordingly, merge the sentence whitespaces
Change mention span offsets from local to global.


## Cross-doc Coref and Question/Context mention Grounding
High-level intuition:
For each of the different types of mentions, figure out a normalization to generate entities and so that different mentions can be resolved to the same entity.

For eg. For entity mentions, mention_surface can be a normalization, for Date mentions the norm-val can be it.
Therefore, all date_mentions that normalize to the same value will be considered mentions of the same date_entity 

For each type:
1. Perform CDCR in context paragraphs to figure set of entities. These are represented as list of mentions that refer to them.
2. Also, for the mention list, generate a new list that stores the entity_idx it is a mention of from [0, ..., num_type_entities]

Main function ```exactStringMatchCDCR``` and ```normalizableMensCDCR```

Following fields are added:
1. ```constants.context_eqent2entmens``` and equivalent versions for NUM and DATE.
Stored as List[List] where outer is the list of entities and the inner are its mentions as (context_idx, mention_idx)   
2. ```constants.context_entmens2entidx``` and equivalent versions for NUM and DATE.
Stored as List[List], same as the list of mentions, only that these contain the entity_idx grounding.
entity_idx goes from ```[0  ... num_entities_of_type_T]```

Question Entity_mentions are also grounded in ```q_entmens2entidx ```.
A list the size of question's entity mention list containing entity_idx.
*Note: If the mention cannot be grounded, -1 is stored as it's grounding* 


## Answer Grounding and Typing --- ```ans_grounding```
For each QA, type the answer in one of 5 types:
```STRING```, ```ENTITY```, ```BOOL```, ```DATE```, or ```NUM```.

Find the best scoring mentions.
If the F1 exceeds the input ```F1-Threshold```, then assign the mention's type, else ```STRING```.

Also ground the answer: 
1. ```STRING```: List of ```(context_idx, (start, end))``` which represent the start and end of the answer span.
Some String answers are not found, instead use ```constants.NO_ANS_GROUNDING``` for such cases.

2. ```ENTITY```, ```DATE```, and ```NUM```: Binary vector (List) containing entity_grounding for the answer.
For each mention that is in the best found mentions, it's corresponding entity is stored as a possible answer.

3. ```BOOL```: 1 (true) or 0 (false).

To store the typing and grounding, the following fields are added:
```constants.ans_type_field``` and ```constants.ans_grounding_field```







 


