# Passage id
passage_id = "passage_id"

# List of QA pair dicts
qa_pairs = "qa_pairs"

# Question
question = "question"
original_question = "original_question"
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
original_passage = "original_passage"
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

# List of (string, start, end(ex), type) tuples for questions
q_num_mens = question + "_" + NUM_TYPE + "_mens"
q_date_mens = question + "_" + DATE_TYPE + "_mens"
# Same size as q_num / q_date -- but idx grounding into equivalent nums / dates in num_entities/date_entities
q_num_entidx = question + "_" + NUM_TYPE + "_men2entidx"
q_date_entidx = question + "_" + DATE_TYPE + "_men2entidx"
# List of normalized values of num/date entities in the idx order
q_num_normalized_values = question + "_" + NUM_TYPE + "_normalized_values"
q_date_normalized_values = question + "_" + DATE_TYPE + "_normalized_values"


# List of (string, token_idx, normalized_vlaue) tuples for passage --- tokenidx = (start, end) for dates
passage_num_mens = passage + "_" + NUM_TYPE + "_mens"
passage_date_mens = passage + "_" + DATE_TYPE + "_mens"
# Same size as passage_{num, date}_mens -- but idx grounding into equivalent numbers/dates
passage_num_entidx = passage + "_" + NUM_TYPE + "_men2entidx"
passage_date_entidx = passage + "_" + DATE_TYPE + "_men2entidx"
# List of normalized values of num/date entities in the idx order
passage_num_normalized_values = passage + "_" + NUM_TYPE + "_normalized_values"
passage_date_normalized_values = passage + "_" + DATE_TYPE + "_normalized_values"


COMMA="@COMMA@"
LRB="@LRB@"
RRB="@RRB@"




