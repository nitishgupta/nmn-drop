from typing import List, Union, Dict

import json
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import utils.util as myutils

import unicodedata
from utils import util, spacyutils
from allennlp.data.tokenizers import Token
from datasets.drop.preprocess import ner_process

from allennlp.tools.squad_eval import metric_max_over_ground_truths
from allennlp.tools.drop_eval import get_metrics as drop_em_and_f1, answer_json_to_strings

from allennlp_semparse.common.util import lisp_to_nested_expression


def f1metric(prediction: Union[str, List], ground_truths: List):  # type: ignore
    """
    Parameters
    ----------a
    prediction: ``Union[str, List]``
        The predicted answer from the model evaluated. This could be a string, or a list of string
        when multiple spans are predicted as answer.
    ground_truths: ``List``
        All the ground truth answer annotations.
    """
    # If you wanted to split this out by answer type, you could look at [1] here and group by
    # that, instead of only keeping [0].
    ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
    exact_match, f1_score = metric_max_over_ground_truths(drop_em_and_f1, prediction, ground_truth_answer_strings)

    return (exact_match, f1_score)


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
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
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


@Predictor.register("drop_demo_predictor")
class DROPDemoPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.spacy_nlp = spacyutils.getSpacyNLP()
        self.spacy_whitespacetokenizer = spacyutils.getWhiteTokenizerSpacyNLP()

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]

        spacy_nlp = self.spacy_nlp
        spacy_whitespacetokenizer = self.spacy_whitespacetokenizer

        # From datasets.drop.preprocess.tokenize
        # Parse passage
        cleaned_passage_text = unicodedata.normalize("NFKD", passage_text)
        cleaned_passage_text = util.pruneMultipleSpaces(cleaned_passage_text)
        passage_spacydoc = spacyutils.getSpacyDoc(cleaned_passage_text, spacy_nlp)
        passage_tokens = [t for t in passage_spacydoc]
        passage_tokens: List[Token] = split_tokens_by_hyphen(passage_tokens)

        passage_token_charidxs = [token.idx for token in passage_tokens]
        passage_token_texts: List[str] = [t.text for t in passage_tokens]
        # Remaking the doc for running NER on new tokenization
        new_passage_doc = spacyutils.getSpacyDoc(" ".join(passage_token_texts), spacy_whitespacetokenizer)

        assert len(passage_tokens) == len(" ".join(passage_token_texts).split(" "))
        assert len(new_passage_doc) == len(passage_tokens)

        # List[Tuple[int, int]] -- start (inclusive) and end (exclusive) token idxs for sentence boundaries
        passage_sent_idxs = sorted([(sentence.start, sentence.end) for sentence in new_passage_doc.sents],
                                   key=lambda x: x[0])
        passage_ners = spacyutils.getNER(new_passage_doc)

        (p_parsed_dates, p_normalized_date_idxs,
         p_normalized_date_values, _) = ner_process.parseDateNERS(passage_ners, passage_token_texts)
        (p_parsed_nums, p_normalized_num_idxs,
         p_normalized_number_values, _) = ner_process.parseNumNERS(passage_ners, passage_token_texts)

        # Parse question
        question: str = question_text.strip()
        cleaned_question = unicodedata.normalize("NFKD", question)
        cleaned_question = util.pruneMultipleSpaces(cleaned_question)

        q_spacydoc = spacyutils.getSpacyDoc(cleaned_question, spacy_nlp)
        question_tokens = [t for t in q_spacydoc]
        question_tokens = split_tokens_by_hyphen(question_tokens)
        question_token_charidxs = [token.idx for token in question_tokens]
        question_token_texts = [t.text for t in question_tokens]

        # Remaking the doc for running NER on new tokenization
        new_question_doc = spacyutils.getSpacyDoc(" ".join(question_token_texts), spacy_whitespacetokenizer)
        assert len(new_question_doc) == len(question_tokens)

        q_ners = spacyutils.getNER(new_question_doc)
        (q_parsed_dates, q_normalized_date_idxs,
         q_normalized_date_values, q_num_date_entities) = ner_process.parseDateNERS(q_ners, question_token_texts)
        (q_parsed_nums, q_normalized_num_idxs,
         q_normalized_number_values, q_num_num_entities) = ner_process.parseNumNERS(q_ners, question_token_texts)

        return self._dataset_reader.text_to_instance(
            question_text=" ".join(question_token_texts),
            original_ques_text=question_text,
            question_charidxs=question_token_charidxs,
            passage_text=" ".join(passage_token_texts),
            original_passage_text=passage_text,
            passage_charidxs=passage_token_charidxs,
            p_sent_boundaries=passage_sent_idxs,
            p_date_mens=p_parsed_dates,
            p_date_entidxs=p_normalized_date_idxs,
            p_date_normvals=p_normalized_date_values,
            p_num_mens=p_parsed_nums,
            p_num_entidxs=p_normalized_num_idxs,
            p_num_normvals=p_normalized_number_values,
            qtype="UNK",
            program_supervised=False,
            qattn_supervised=False,
            execution_supervised=False,
            pattn_supervised=False,
            strongly_supervised=False,
            ques_attn_supervision=None,
            date_grounding_supervision=None,
            num_grounding_supervision=None,
            passage_attn_supervision=None,
            synthetic_numground_metadata=None,
            answer_passage_spans=None,
            answer_question_spans=None,
            question_id="demo_question",
            passage_id="demo_passage",
            answer_annotations=None,
            max_question_len=50,
        )


    def convert_wordpiece_attention_to_tokens(self, wp_attention, tokens, wps, wpidx2tokenidx):
        tokens_len = len(tokens)
        wps_len = len(wps)
        wp_attention = wp_attention[:wps_len]  # attention over word-pieces
        token_attention = [0.0] * tokens_len   # attention over tokens
        for token_idx, attn_value in zip(wpidx2tokenidx, wp_attention):
            if token_idx >= 0 and token_idx < tokens_len:
                token_attention[token_idx] += attn_value
        return token_attention


    def get_module_name_mapping(self, module_name):
        module_name_mapping = {
            "find_PassageAttention": "find",
            "filter_PassageAttention": "filter",
            "relocate_PassageAttention": "relocate",
            "compare_date_lesser_than": "compare-date-lt",
            "compare_date_greater_than": "compare-date-gt",
            "compare_num_lesser_than": "compare_num_lt",
            "compare_num_greater_than": "compare_num_gt",
            "year_difference": "year-diff",
            "year_difference_single_event": "year_difference",
            "find_passageSpanAnswer": "span",
            "passageAttn2Count": "count",
            "find_PassageNumber": "find-num",
            "minNumPattn": "find-min-num",
            "maxNumPattn": "find-max-num",
        }

        return module_name_mapping.get(module_name, module_name)


    def rename_modules_in_nested_expression(self, nested_expression):
        mapped_expression = []
        for i, argument in enumerate(nested_expression):
            if isinstance(argument, str):
                mapped_expression.append(self.get_module_name_mapping(argument))
            elif isinstance(argument, list):
                mapped_expression.append(self.rename_modules_in_nested_expression(argument))
            else:
                raise NotImplementedError
        return mapped_expression

    def nested_expression_to_lisp(self, nested_expression):
        if isinstance(nested_expression, str):
            return nested_expression

        elif isinstance(nested_expression, List):
            lisp_expressions = [self.nested_expression_to_lisp(x) for x in nested_expression]
            return "(" + " ".join(lisp_expressions) + ")"
        else:
            raise NotImplementedError

    def print_execution_for_debugging(self, program_execution, passage_tokens,  question_tokens,
                                      num_values, date_values):
        for module_dict in program_execution:
            for module, infos_dict in module_dict.items():
                # infos : Tuple["type", attention]
                print(f"{module}: {[x for x in infos_dict]} ")
                for output_type, attention in infos_dict.items():
                    output_type = output_type.split("_")[0]
                    if output_type == "passage":
                        # passage_attention = self.convert_wordpiece_attention_to_tokens(attention, passage_tokens,
                        #                                                                passage_wps,
                        #                                                                passage_wpidx2tokenidx)
                        print([(i, j) for (i, j) in zip(passage_tokens, attention)])
                    elif output_type == "question":
                        # question_attention = self.convert_wordpiece_attention_to_tokens(attention, question_tokens,
                        #                                                                 question_wps,
                        #                                                                 question_wpidx2tokenidx)
                        print([(i, j) for (i, j) in zip(question_tokens, attention)])
                    elif output_type == "number":
                        print([(i, j) for (i, j) in zip(num_values, attention)])
                    elif output_type == "date":
                        print([(i, j) for (i, j) in zip(date_values, attention)])
                    else:
                        pass


    def convert_execution_attention_to_tokens(self, program_execution, passage_tokens, passage_wps,
                                              passage_wpidx2tokenidx, question_tokens, question_wps,
                                              question_wpidx2tokenidx):
        """
        program_execution is a list of dicts, i.e. [{"module_name": Dict}], where each inner dict is {"type", attention}

        This function converts attention for "question" and "passage" types from attention over word-pieces to tokens.
        """
        for module_dict in program_execution:
            for module, infos_dict in module_dict.items():
                for output_type, attention in infos_dict.items():

                    output_type_canonical = output_type.split("_")[0]
                    if output_type_canonical == "passage":
                        passage_attention = self.convert_wordpiece_attention_to_tokens(attention, passage_tokens,
                                                                                       passage_wps,
                                                                                       passage_wpidx2tokenidx)
                        infos_dict[output_type] = passage_attention
                    elif output_type_canonical == "question":
                        question_attention = self.convert_wordpiece_attention_to_tokens(attention, question_tokens,
                                                                                        question_wps,
                                                                                        question_wpidx2tokenidx)
                        infos_dict[output_type] = question_attention
        return program_execution



    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        outputs = self.predict_instance(instance)
        logical_programs = outputs["batch_logical_programs"]
        metadata = outputs["metadata"]
        question = metadata["original_question"]
        passage = metadata["original_passage"]
        date_values = metadata["passage_date_values"]
        num_values = metadata["passage_number_values"]
        year_diff_values = metadata["passage_year_diffs"]
        passage_mask = outputs["passage_mask"]

        question_tokens = metadata["question_orig_tokens"]
        question_wps = metadata["question_tokens"]
        question_wpidx2tokenidx = metadata["question_wpidx2tokenidx"]

        # Last wordpiece is '[SEP]'
        passage_tokens = metadata["passage_orig_tokens"]
        passage_wps = metadata["passage_tokens"]
        passage_wpidx2tokenidx = metadata["passage_wpidx2tokenidx"]

        predicted_ans = outputs["predicted_answer"]

        program_nested_expressions = [lisp_to_nested_expression(program) for program in logical_programs]
        program_nested_expression = program_nested_expressions[0]
        # Mapping nested expression's modules to module-names used in the paper
        program_nested_expression = self.rename_modules_in_nested_expression(program_nested_expression)
        program_lisp = self.nested_expression_to_lisp(program_nested_expression)

        # Is a list which contains for each program executed for this instance, its module execution info.
        # Since programs are sorted in decreasing order of score, the first element in the list should be argmax program
        modules_debug_infos = outputs["modules_debug_infos"]
        # Each program execution is linearized in the order that the modules were executed.
        # This execution is a list of dicts, i.e. [{"module_name": Dict}] where each dict contains
        # different outputs a module produces. Each output is a dict itself {"type", attention}, where "type"
        # indicates the type of support the attention is produced over. E.g. "paragraph", "question", "number", etc.
        # If two or more attentions are produced over the same type (e.g. num-compare-lt produces two number attns),
        # the type is differentiated by appending _N (underscore number). E.g. "number_1", "number_2", etc.
        program_execution: List[Dict] = modules_debug_infos[0]
        # Aggregate attention predicted over word-pieces to attention over tokens
        program_execution = self.convert_execution_attention_to_tokens(program_execution, passage_tokens, passage_wps,
                                                                       passage_wpidx2tokenidx, question_tokens,
                                                                       question_wps, question_wpidx2tokenidx)

        # print(program_lisp)
        # print(program_nested_expression)
        # self.print_execution_for_debugging(program_execution, passage_tokens, question_tokens,
        #                                    num_values, date_values)

        output_dict = {
            "question": question,
            "passage": passage,
            "predicted_ans": predicted_ans,
            "answer": predicted_ans,
            "question_tokens": question_tokens,
            "passage_tokens": passage_tokens,
            "numbers": num_values,
            "dates": date_values,
            "year_diff_values": year_diff_values,
            "program_nested_expression": program_nested_expression,
            "program_lisp": program_lisp,
            "program_execution": program_execution,
        }
        return output_dict

    def _print_ExecutionValTree(self, exval_tree, depth=0):
        """
        exval_tree: [[root_func_name, value], [], [], []]
        """
        tabs = "\t" * depth
        func_name = str(exval_tree[0][0])
        debug_value = str(exval_tree[0][1])
        debug_value = debug_value.replace("\n", "\n" + tabs)
        outstr = f"{tabs}{func_name}  :\n {tabs}{debug_value}\n"
        if len(exval_tree) > 1:
            for child in exval_tree[1:]:
                outstr += self._print_ExecutionValTree(child, depth + 1)
        return outstr


