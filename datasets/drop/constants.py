# Passage id
passage_id = "passage_id"

# List of QA pair dicts
qa_pairs = "qa_pairs"

# Question
question = "question"
question_tokens = "question_tokens"
question_charidxs = "question_charidxs"
cleaned_question = "cleaned_question"       # DEPRECATED
tokenized_question = "tokenized_question"   # DEPRECATED
# Answer dict
answer = "answer"
# Number answer as a string
num_ans = "number"
# Dict with "day", "month", "year" as field
date_ans = "date"
# List of spans
spans_ans = "spans"

# query id
query_id = "query_id"

# Passage text
passage = "passage"
passage_tokens = "passage_tokens"
passage_charidxs = "passage_charidxs"
passage_sent_idxs = "passage_sent_idxs"

cleaned_passage = "cleaned_passage"         # DEPRECATED
tokenized_passage = "tokenized_passage"     # DEPRECATED

# Answer string
# Original and tokenized
ans_field = "answer"

answer_passage_spans = "answer_passage_spans"
answer_question_spans = "answer_question_spans"

answer_type = "answer_type"

# This bool key is added to QA dict if it is an augmented QA example
augmented_example = "augmented_example"

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
qtype = "qtype"
strongly_supervised = "strongly_supervised"
program_supervised = "program_supervised"
qattn_supervised = "qattn_supervised"
execution_supervised = "execution_supervised"
pattn_supervised = "pattn_supervised"  # Boolean
program_supervision = "program_supervision"

# n-tuple of question attention
ques_attention_supervision = "ques_attention_supervision"

# Supervision keys
question_attention_supervision = "question_attention_supervision"
date_entidxs = "date_entidxs"
num_entidxs = "num_entidxs"
shared_substructure_annotations = "shared_substructure_annotations"


# Date comparision question-type
DATECOMP_QTYPE = "date_comparison"
qspan_dategrounding_supervision = "qspan_dategrounding_supervision"
qspan_datevalue_supervision = "qspan_datevalue_supervision"

# Number comparision questions -- grounding should be a 2-tuple of grounding into passage_num_normalized_values
NUMCOMP_QTYPE = "number_comparison"
qspan_numgrounding_supervision = "qspan_numgrounding_supervision"
qspan_numvalue_supervision = "qspan_numvalue_supervision"

# How many yards was the longest/shortest ..
YARDS_longest_qtype = "how_many_yards_longest"
YARDS_shortest_qtype = "how_many_yards_shortest"
YARDS_second_longest_qtype = "how_many_yards_second_longest"
YARDS_second_shortest_qtype = "how_many_yards_second_shortest"
# How many yards -- Find a PassageNum as answer
YARDS_findnum_qtype = "how_many_yards_findnum"

MAX_find_qtype = "max_find_qtype"
MIN_find_qtype = "min_find_qtype"
NUM_find_qtype = "num_find_qtype"
MAX_filter_find_qtype = "max_filterfind_qtype"
MIN_filter_find_qtype = "min_filterfind_qtype"
NUM_filter_find_qtype = "num_filterfind_qtype"

# Difference between two passage numbers. Each number can be a direct grounding, or maximum / minimum of a set
DIFF_MAXMIN_qtype = "diff_maxmin_qtype"
DIFF_MAXNUM_qtype = "diff_maxnum_qtype"
DIFF_MAXMAX_qtype = "diff_maxmax_qtype"

DIFF_NUMMAX_qtype = "diff_nummax_qtype"
DIFF_NUMMIN_qtype = "diff_nummin_qtype"
DIFF_NUMNUM_qtype = "diff_numnum_qtype"

DIFF_MINMAX_qtype = "diff_minmax_qtype"
DIFF_MINNUM_qtype = "diff_minnum_qtype"
DIFF_MINMIN_qtype = "diff_minmin_qtype"

# Subset of "How many" which require counting over passage attention
COUNT_find_qtype = "count_find_qtype"
COUNT_filter_find_qtype = "count_filterfind_qtype"

# Relocate question prog type
RELOC_find_qtype = "relocate_find_qtype"
RELOC_filterfind_qtype = "relocate_filterfind_qtype"
RELOC_maxfind_qtype = "relocate_maxfind_qtype"
RELOC_maxfilterfind_qtype = "relocate_maxfilterfind_qtype"
RELOC_minfind_qtype = "relocate_minfind_qtype"
RELOC_minfilterfind_qtype = "relocate_minfilterfind_qtype"

# YEAR Diff question types
YEARDIFF_SE_qtype = "yeardiff_find_qtype"  # Single event
YEARDIFF_TE_qtype = "yeardiff_find2_qtype"  # Two Events

# Synthetic questions
passage_attn_supervision = "passage_attn_supervision"  # passage-attention


SYN_COUNT_qtype = "synthetic_count_qtype"
SYN_NUMGROUND_qtype = "synthetic_numground_qtype"  # Synthetic questions for number grounding
SYN_NUMGROUND_METADATA = "synthetic_numground_metadata"  # Synthetic num ques come with Metadata

COMMA = "@COMMA@"
LRB = "@LRB@"
RRB = "@RRB@"
