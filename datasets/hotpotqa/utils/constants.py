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

# TYPES
NUM_TYPE = "NUM"
DATE_TYPE = "DATE"
ENTITY_TYPE = "ENTITY"
BOOL_TYPE = "BOOL"
STRING_TYPE = "STRING"

# List of NERs
q_ner_field = q_field + "_ner"
q_ent_ner_field = q_field + "_ner_" + ENTITY_TYPE
q_num_ner_field = q_field + "_ner_" + NUM_TYPE
q_date_ner_field = q_field + "_ner_" + DATE_TYPE

ans_ner_field = ans_field + "_ner"

# List of paras, each with list of sentences, each with a list of NERs
context_ner_field = context_field + "_ner"
context_ent_ner_field = context_field + "_ner_" + ENTITY_TYPE
context_num_ner_field = context_field + "_ner_" + NUM_TYPE
context_date_ner_field = context_field + "_ner_" + DATE_TYPE

dates_normalized_field = "dates_normalized"
nums_normalized_field = "nums_normalized"

# These are like entities, for different types of mentions
# Entity mentions that are coreferred, number mentions and date mentions that have the same denotation
# List of list: Outer list size is the num of ents; inner list indexes into the corresponding context_TYPE_ner_field
context_eqent2entmens = "context_eqent2entmens"
context_eqent2nummens = "context_eqent2nummens"
context_eqent2datemens = "context_eqent2datemens"

# List mapping mentions to the corresponding entities in the CONTEXT_EQTYPE_TO_MENS
context_entmens2entidx = "context_entmens2entidx"
context_nummens2entidx = "context_nummens2entidx"
context_datemens2entidx = "context_datemens2entidx"

# Grounding entity_type mens in the question to context entities.
# List of size as q_ent_ner_field
q_entmens2entidx = "q_entmens2entidx"

# To store the predicted answer
pred_ans = "pred_ans"

# To specify no link between Q-mention and context entities
NO_LINK = "@NOLINK"

NO_ANS_GROUNDING = "@NO_ANS_GROUNDING"

'''
# Dict from entity id to list of (context_id, mention_id)
ENT_TO_CONTEXT_MENS = "entity2context_mens"
# (List of list) For every context, list of entity ids for each mention
CONTEXT_MENS_TO_ENT = "context_mens_entlinks"
# List of grounding for Ques mentions
Q_MENS_TO_ENT = "q_mens_entlinks"
'''




