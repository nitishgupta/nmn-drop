# DROP pre-processing

## Tokenize, answer_spans, num, and date parsing
Script tokenize.py handles all preprocessing - 
1. Uses spacy to tokenize - replaces passage and questions with tokenized strs
2. Use code (copied) from allennlp for answer span finding - only span answers supported right now
    1. answer_passage_spans / answer_question_spans - List of (start, end) spans for the answers
    2. answer_type - One of NUM_TYPE, DATE_TYPE, SPAN_TYPE
3. Parsing number and dates:
*Each passage also has the the fields listed below.*
    1. q_{date, num}_mens - List of (str, (start, end), normalized_val) - num only has a token_idx instead of the start/end
    2. q_{date, num}_entidx  - List (same len as 1.) of date/num entity_idxs i.e. mens that resolve to equal normalized values 
    3. q_{date, num}_normalized_values - List of normalized values of num/date entities in the order of entity_idxs in 2.   
    
## Prune questions 
Number and Date based pruning of questions - script prune_ques.py