# HotPotQA Dataset

# Directory structure
### raw
Contains original json files from the dataset release.

Keys for json are in ```utils.constants```.

### tokenized
Contains tokenized and NER question, answer and context sentences

Script used: ```preprocess.tokenize``` with ```propn = False```.
Tokenized sentences are stored with space-delimition, and NERs with additonal keys as in ```utils.constants```.

### tokenized_propn
Same as ```tokenized``` only difference being, all ProperNoun spans that do not intersect with NERs are also labeled as NER.



