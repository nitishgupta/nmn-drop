from typing import List, Tuple, Dict, Union, Callable
import random
from dataclasses import dataclass

from allennlp.data import Token
from allennlp.data.tokenizers import SpacyTokenizer

from semqa.utils.qdmr_utils import node_from_dict, Node, lisp_to_nested_expression, \
    nested_expression_to_tree


random.seed(42)

WHTOKENS = ["who", "where", "when", "how", "whom", "which", "what", "whose", "why"]
WH_PHRASE_LABELS = ["WHNP", "WHADVP", "WP", "WHPP", "WRB", "WDT", "WHADJP"]
dep_token_idx = "token_index"
dep_node_type = "nodeType"
dep_link = "link"
dep_children = "children"

@dataclass
class Question:
    qid: str
    question_str: str
    tokens: List[str]
    pos: List[str]
    conparse_spans: List   # List[(start, end(ex), text, LABEL)]
    depparse_tree: Dict     # {"nodeType", "link", "token_index", "children": List[Dict]}
    length: int



def map_question_parse(squad_questions: List[Dict],
                       squad_questions_conparse: List[Dict],
                       squad_questions_depparse: List[Dict]) -> Dict[str, Question]:
    """Return a mapping from query-id to Question"""
    assert len(squad_questions) == len(squad_questions_conparse), print(f"Num of ques and con-parse is not equal."
                                                                        f" {len(squad_questions)}"
                                                                        f" != {len(squad_questions_conparse)}")
    assert len(squad_questions) == len(squad_questions_depparse), print(f"Num of ques and dep-parse is not equal."
                                                                        f" {len(squad_questions)}"
                                                                        f" != {len(squad_questions_depparse)}")

    print("Num of input questions: {}".format(len(squad_questions)))
    qid2question = {}
    for qdict, conparse_dict, depparse_dict in zip(squad_questions, squad_questions_conparse, squad_questions_depparse):
        question = qdict["sentence"]
        qid = qdict["sentence_id"]
        qtokens = conparse_dict["tokens"]
        conparse_spans: List = conparse_dict["spans"]
        pos = depparse_dict["pos"]
        depparse_tree: Dict = depparse_dict["hierplane_tree"]["root"]
        assert len(qtokens) == len(pos)
        num_tokens = len(qtokens)


        qobj: Question = Question(qid=qid, question_str=question, tokens=qtokens, pos=pos, conparse_spans=conparse_spans,
                                  depparse_tree=depparse_tree, length=num_tokens)
        qid2question[qid] = qobj
    return qid2question



def get_wh_phrase(question: Question):
    return question.conparse_spans[1]


def is_firstspan_whphrase(question: Question):
    wh_span = get_wh_phrase(question)
    start, end, text, label = wh_span
    return label in WH_PHRASE_LABELS



def tokenize(text: str, spacy_tokenizer:SpacyTokenizer) -> List[Token]:
    tokens: List[Token] = spacy_tokenizer.tokenize(text)
    tokens = split_tokens_by_hyphen(tokens)
    return tokens


def _split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += _split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def add_project_select_program_supervision(qa_dict: Dict):
    """Add project(select) program-supervision to a DROP style qa-dict. """
    # Adding simple project(select) program annotation without question-attention supervision
    program_supervision_lisp = "(select_passagespan_answer (project_passage select_passage))"
    nested_expr = lisp_to_nested_expression(program_supervision_lisp)
    program_node: Node = nested_expression_to_tree(nested_expr)

    program_supervision = program_node.to_dict()
    qa_dict["program_supervision"] = program_supervision
    return qa_dict


def make_qa_pair_dict(qid: str, question: str, answer_texts: List[str], spacy_tokenizer):
    """Structure of DROP data:

    {
        "para_id": {
            "passage": passage-text,
            "qa_pairs": [
                {
                    "question": ...,
                    "answer": {"number": "", "date": {"day":"", "month": "", "year": ""}, "spans":[]},
                    "query_id": qid,
                    "highlights": [],
                    "question_type": [],
                    "validated_answers": List["answer"-dict],
                    "expert_answers": [],
                    "question_tokens": [token, ....],
                    "question_charidxs": [ .... ],
                    "question_DATE_mens": [],
                    "question_DATE_men2entidx": [],
                    "question_DATE_normalized_values": [],
                    "question_NUM_mens": [],
                    "question_NUM_men2entidx": [],
                    "question_NUM_normalized_values": [],
                    "answer_passage_spans": [],
                    "answer_question_spans": [],
                    "program_supervision": node_to_dict,
                }
            ],
            "passage_tokens": [token, ...],
            "passage_charidxs": [charidx, ...],
            "passage_sent_idxs": [],
            "passage_DATE_mens": [],
            "passage_DATE_men2entidx": [],
            "passage_DATE_normalized_values": [],
            "passage_NUM_mens": [],
            "passage_NUM_men2entidx": [],
            "passage_NUM_normalized_values": []
        }
    }
    """
    q_spacy_tokens: List[Token] = tokenize(question, spacy_tokenizer)
    q_spacy_tokens_texts: List[str] = [t.text for t in q_spacy_tokens]
    ques_token_charidxs: List[int] = [token.idx for token in q_spacy_tokens]

    # # Checking if already supplied question tokens are equivalent to the re-tokenization performed here.
    # # If equal; use the token-charidxs from above, otherwise ...
    # if q_spacy_tokens_texts != ques_tokens:
    #     print("Pre-tokenization and current-tokenization are not the same")
    #     print(f"ques:{question}  pre-token:{ques_tokens}  currrent-tokens:{q_spacy_tokens_texts}")

    answer_dict = {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [answer_texts[0]]}
    validated_answers = []
    if len(answer_texts) > 1:
        for i in range(1, len(answer_texts)):
            val_answer_dict = {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [answer_texts[i]]}
            validated_answers.append(val_answer_dict)

    qa_pair_dict = {
        "question": question,
        "query_id": qid,
        "answer": answer_dict,
        "highlights": [],
        "question_type": [],
        "validated_answers": validated_answers,
        "expert_answers": [],
        "question_tokens": q_spacy_tokens_texts,
        "question_charidxs": ques_token_charidxs,
        "question_DATE_mens": [],
        "question_DATE_men2entidx": [],
        "question_DATE_normalized_values": [],
        "question_NUM_mens": [],
        "question_NUM_men2entidx": [],
        "question_NUM_normalized_values": [],
        "answer_passage_spans": [],     # this should be handled by the reader
        "answer_question_spans": [],    # this should be handled by the reader
    }

    # # Make program-supervision from ques_parse: List[(start_idx, end_idx (_exclusive_), "tokenized_span_text", "LABEL")]
    # relevant_span_labels = ["WHNP", "WHADVP", "WP", "WHPP", "WRB", "WDT", "WHADJP"]
    # first_span = ques_parse[1]  # skipping full sentence span
    #
    # project_start_idx, project_end_idx = None, None
    # select_start_idx, select_end_idx = None, None
    #
    # if first_span[3] in relevant_span_labels and first_span[0] == 0:
    #     project_start_idx = first_span[0]
    #     project_end_idx = first_span[1] - 1
    #
    # # for span in ques_parse:
    # #     if span[3] in relevant_span_labels and span[0] == 0:
    # #         project_start_idx = span[0]
    # #         project_end_idx = span[1] - 1  # _inclusive_
    # #         break
    #
    # if project_start_idx is not None:
    #     select_start_idx = project_end_idx + 1
    #     select_end_idx = len(ques_tokens) - 2  # skip last token == ?
    # else:
    #     return None
    # project_start_idx: int = project_start_idx
    # project_end_idx: int = project_end_idx
    # select_start_idx: int = select_start_idx
    # select_end_idx: int = select_end_idx
    #
    # project_node = Node(predicate="project_passage",
    #                     string_arg=" ".join(ques_tokens[project_start_idx:project_end_idx + 1]))
    # project_ques_attention = [1 if project_start_idx <= i <= project_end_idx else 0 for i in range(len(ques_tokens))]
    # project_node.supervision["question_attention_supervision"] = project_ques_attention
    #
    # select_node = Node(predicate="select_passage",
    #                    string_arg=" ".join(ques_tokens[select_start_idx:select_end_idx + 1]))
    # select_ques_attention = [1 if select_start_idx <= i <= select_end_idx else 0 for i in range(len(ques_tokens))]
    # select_node.supervision["question_attention_supervision"] = select_ques_attention
    #
    # project_node.add_child(select_node)
    #
    # program_node = Node(predicate="select_passagespan_answer")
    # program_node.add_child(project_node)
    #
    # program_node.extras["constituency_parse_spans"] = ques_parse
    # program_node.extras["project_start_end_indices"] = (project_start_idx, project_end_idx)
    # # program_node.extras["select_start_end_indices"] = (select_start_idx, select_end_idx)
    #
    # program_supervision = program_node.to_dict()
    # qa_pair_dict["program_supervision"] = program_supervision

    return qa_pair_dict

