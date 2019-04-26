# Passage id
passage_id = "passage_id"

# List of QA pair dicts
qa_pairs = "qa_pairs"

# Question
question = "question"
cleaned_question = "cleaned_question"
tokenized_question = "tokenized_question"
question_charidxs = "question_charidxs"
# Answer dict
answer = "answer"
# Number answer as a string
num_ans = "number"
# Dict with "day", "month", "year" as field
date_ans = "date"
# List of spans
spans_ans = "spans"

#query id
query_id = "query_id"

# Passage text
passage = "passage"
cleaned_passage = "cleaned_passage"
tokenized_passage = "tokenized_passage"
passage_charidxs = "passage_charidxs"

# Answer string
# Original and tokenized
ans_field = "answer"

answer_passage_spans = "answer_passage_spans"
answer_question_spans = "answer_question_spans"

answer_type = "answer_type"

# TYPES
NUM_TYPE = "NUM"
DATE_TYPE = "DATE"
SPAN_TYPE = "SPAN"
UNK_TYPE = "UNKNOWN_TYPE"

# List of (string, start, end(ex), type) tuples for questions
q_num_mens = "question_" + NUM_TYPE + "_mens"
q_date_mens = "question_" + DATE_TYPE + "_mens"
# Same size as q_num / q_date -- but idx grounding into equivalent nums / dates in num_entities/date_entities
q_num_entidx = "question_" + NUM_TYPE + "_men2entidx"
q_date_entidx = "question_" + DATE_TYPE + "_men2entidx"
# List of normalized values of num/date entities in the idx order
q_num_normalized_values = "question_" + NUM_TYPE + "_normalized_values"
q_date_normalized_values = "question_" + DATE_TYPE + "_normalized_values"


# List of (string, token_idx, normalized_vlaue) tuples for passage --- tokenidx = (start, end) for dates
passage_num_mens = "passage_" + NUM_TYPE + "_mens"
passage_date_mens = "passage_" + DATE_TYPE + "_mens"
# Same size as passage_{num, date}_mens -- but idx grounding into equivalent numbers/dates
passage_num_entidx = "passage_" + NUM_TYPE + "_men2entidx"
passage_date_entidx = "passage_" + DATE_TYPE + "_men2entidx"
# List of normalized values of num/date entities in the idx order
passage_num_normalized_values = "passage_" + NUM_TYPE + "_normalized_values"
passage_date_normalized_values = "passage_" + DATE_TYPE + "_normalized_values"


# STRONG SUPERVISION FIELDS
# Boolean
strongly_supervised = "strongly_supervised"
# String
qtype = "qtype"
# n-tuple of question attention
ques_attention_supervision = "ques_attention_supervision"

# Date comparision question-type
DATECOMP_QTYPE = "date_comparison"
qspan_dategrounding_supervision = "qspan_dategrounding_supervision"
qspan_datevalue_supervision = "qspan_datevalue_supervision"

# Number comparision questions -- grounding should be a 2-tuple of grounding into passage_num_normalized_values
NUMCOMP_QTYPE = "number_comparison"
qspan_numgrounding_supervision = "qspan_numgrounding_supervision"
qspan_numvalue_supervision = "qspan_numvalue_supervision"


# How many yards was the longest/shortest ..
YARDS_longest_qtype = 'how_many_yards_longest'
YARDS_shortest_qtype = 'how_many_yards_shortest'
YARDS_second_longest_qtype = 'how_many_yards_second_longest'
YARDS_second_shortest_qtype = 'how_many_yards_second_shortest'

COMMA="@COMMA@"
LRB="@LRB@"
RRB="@RRB@"




