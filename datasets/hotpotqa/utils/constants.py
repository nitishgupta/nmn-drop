# Originally from the dataset
id_field = "_id"

# Tokenized (space-delim) question
q_field = "question"

# Answer string
# Original and tokenized
ans_field = "answer"

# List of (title, sent_id) tuples
suppfacts_field = "supporting_facts"

# Tokenized (space-delim) context. List of paras, each with list of sentences
context_field = "context"

qtyte_field = "type"    # comparision or bridge

qlevel_field = "level"  # easy, medium, or hard

# Added by the user'
q_whitespaces_field = "q_whitespaces"
context_whitespaces_field = "context_whitespaces"


ans_tokenized_field = "answer_tokenized"
ans_type_field = "answer_type"
ans_grounding_field = "answer_grounding"

# List of NERs
q_ner_field = q_field + "_ner"
ans_ner_field = ans_field + "_ner"
# List of paras, each with list of sentences, each with a list of NERs
context_ner_field = context_field + "_ner"

dates_normalized_field = "dates_normalized"
nums_normalized_field = "nums_normalized"

# TYPES
NUM_TYPE = "NUM"
DATE_TYPE = "DATE"
ENTITY_TYPE = "ENTITY"
BOOL_TYPE = "BOOL"
STRING_TYPE = "STRING"


# To store the predicted answer
pred_ans = "pred_ans"

# To specify no link between Q-mention and context entities
NO_LINK = "@NOLINK"

# Dict from entity id to list of (context_id, mention_id)
ENT_TO_CONTEXT_MENS = "entity2context_mens"
# (List of list) For every context, list of entity ids for each mention
CONTEXT_MENS_TO_ENT = "context_mens_entlinks"
# List of grounding for Ques mentions
Q_MENS_TO_ENT = "q_mens_entlinks"




