# Preprocessing HotpotQA

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

Normalize types of NE and values in certain NE, for both contexts and questions

1. For ```UNCHANGED_NER_TYPES``` --- keep the mention as is
2. Remove ```REMOVE_NER_TYPES``` types of mentions
3. For other kinds, normalize to ```NUM``` or ```DATE```
4. Get a ```NUMBER_DICT``` and ```DATE_DICT``` --- contains mappings from string to normalized value.

Apart from changing the NER list, following fields are added.
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
1. Perform CDCR in context paragraphs to figure set of entities. For each entity give a key.
CDCR will only by performed for ENTITY spans, and not NUM and DATE
2. Store for each entity, it's mentions, and for each mention it's entity.
3. Ground mentions in questions to these identified entities, else ground to ```constants.NOLINK```

Main function ```exactStringMatchCDCR``` ---  We implement exact string match coreference

Following fields are added:
1. ```constants.ENT_TO_CONTEXT_MENS```: entity_id --> list of (context_idx, mention_idx)  
2. ```constants.CONTEXT_MENS_TO_ENT```: For each context, a list of entity_ids for the corresponsing mention.
If mention_type is NUM or DATE, then grounding is ```constants.NOLINK```.
List for context is the same length as the number of mentions in it.
3. ```constants.Q_MENS_TO_ENT```: Same as context mention grounding,  a list of groundings for question mention.
Since some question mentions cannot be grounded, ```constants.NOLINK``` is used for their grounding.


## Answer Grounding and Typing --- ```ans_grounding```
For each QA, type the answer in one of 5 types:
```STRING```, ```ENTITY```, ```BOOL```, ```DATE```, or ```NUM```.

Find the best scoring mentions.
If the F1 exceeds the input ```F1-Threshold```, then assign the mention's type, else ```STRING```.

Also ground the answer: 
1. ```STRING```: List of ```(context_idx, (start, end))``` which represent the start and end of the answer span.
Some String answers are not found, instead use ```constants.NOLINK``` for such cases.

2. ```ENTITY```, ```DATE```, and ```BOOL```: List of ```(context_idx, mention_idx)```

3. ```BOOL```: Either 1 (true) or 0 (false).

To store the typing and grounding, the following fields are added:
```constants.ans_type_field``` and ```constants.ans_grounding_field```







 


