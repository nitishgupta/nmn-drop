# Who-did-what

### xmlReader.py --- convert XML input to JSONL files
Keys in output doc:
1. qid
2. contextId
3. contextPara
4. qleftContext
5. qrightContext
6. correctChoice
7. candidateChoices


### tokenize.py  
Tokenize, POS tag, and NP chunk the raw JSONL files.
Splits sentences longer than 120 tokens into smaller chunks.

The NP span offsets for the right ques-context are w.r.t. the full question sentence, including a @BLANK token.


