from typing import List, Union, Dict, Optional

import json
import numpy as np
import torch
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


import unicodedata
from utils import util, spacyutils
from allennlp.data.tokenizers import Token
from datasets.drop.preprocess import ner_process

import allennlp.nn.util as allenutil
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


def print_execution_for_debugging(program_execution, passage_tokens, question_tokens,
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


def convert_wordpiece_attention_to_tokens(wp_attention, tokens, wps, wpidx2tokenidx):
    tokens_len = len(tokens)
    wps_len = len(wps)
    wp_attention = wp_attention[:wps_len]  # attention over word-pieces
    token_attention = [0.0] * tokens_len   # attention over tokens
    for token_idx, attn_value in zip(wpidx2tokenidx, wp_attention):
        if token_idx >= 0 and token_idx < tokens_len:
            token_attention[token_idx] += attn_value
    return token_attention


class Input:
    def __init__(self, name: str, tokens: List):
        self.name: str = name
        self.tokens: List[str] = [str(t) for t in tokens]

    def to_dict(self):
        json_dict = {
            "name": self.name,
            "tokens": self.tokens
        }
        return json_dict


class Output:
    def __init__(self, input_name: str, values: List[float], label: Optional[str] = None):
        self.input_name = input_name
        self.values = values
        self.label = label

    def to_dict(self):
        json_dict = {
            "input_name": self.input_name,
            "values": self.values,
            "label": self.label
        }
        return json_dict


class Module:
    def __init__(self, name:str, identifier: int):
        self.name = name
        self.identifier = identifier

    def to_dict(self):
        json_dict = {
            "name": self.name,
            "identifier": self.identifier
        }
        return json_dict


def add_identifier(nested_expression, count):
    """Convert the nested_expression into a representation that contains the order in which the modules are executed.

    This function converts the nested_expression of module-as-str into expression with module-as-Module class where the
    class stores an `identifier` key which is the number at which the module was executed.

    Since the program-tree is executed in a left-to-right post-traversal order we will traverse the tree in a similar
    manner to number the modules in the nested-expression.
    """
    # If expression is not a list (hence a str) it's a Module
    if not isinstance(nested_expression, list):
        return Module(name=nested_expression, identifier=count), count + 1
    # If expression is tree
    else:
        sub_expression = []
        # Performing left-to-right post traversal of the tree
        for i in range(1, len(nested_expression)):
            arg_i, count = add_identifier(nested_expression[i], count)
            sub_expression.append(arg_i)
        # Then add the root-module of the tree
        arg_0 = Module(name=nested_expression[0], identifier=count)
        sub_expression.insert(0, arg_0)

        return sub_expression, count + 1


def convert_module_expression_tree_to_dict(module_expression):
    mapped_expression = []
    for i, argument in enumerate(module_expression):
        if isinstance(argument, list):
            mapped_expression.append(convert_module_expression_tree_to_dict(argument))
        elif isinstance(argument, Module):
            mapped_expression.append(argument.to_dict())
        else:
            raise NotImplementedError
    return mapped_expression


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
            "year_difference_single_event": "year-diff",
            "find_passageSpanAnswer": "span",
            "passageAttn2Count": "count",
            "find_PassageNumber": "find-num",
            "minNumPattn": "find-min-num",
            "maxNumPattn": "find-max-num",
            "passagenumber_difference": "number-difference",
            "passagenumber_addition": "number-addition",
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
                        passage_attention = convert_wordpiece_attention_to_tokens(attention, passage_tokens,
                                                                                  passage_wps,
                                                                                  passage_wpidx2tokenidx)
                        infos_dict[output_type] = passage_attention
                    elif output_type_canonical == "question":
                        question_attention = convert_wordpiece_attention_to_tokens(attention, question_tokens,
                                                                                   question_wps,
                                                                                   question_wpidx2tokenidx)
                        infos_dict[output_type] = question_attention
        return program_execution


    def convert_module_outputs_to_list(self, program_execution: List[Dict]):
        """Modify the program_execution, which stores each modules output as a dict into a list of Output struct.

        Arguments:
        ----------
        program_execution: ``List[Dict]``
            Each program execution is linearized in the order that the modules were executed.
            This execution is a list of dicts, i.e. [{"module_name": Dict}] where each dict contains
            different outputs a module produces. Each output is a dict itself {"type", attention}, where "type"
            indicates the type of support the attention is produced over. E.g. "paragraph", "question", "number", etc.
            If two or more attentions are produced over the same type (e.g. num-compare-lt produces two number attns),
            the type is differentiated by appending _N (underscore number). E.g. "number_1", "number_2", etc.

        Returns:
        --------
        program_execution: ``List[Dict]``
            [{"module_name": List[Output]}]

        """

        modified_program_execution: List[Dict] = []
        for module_exec_dict in program_execution:
            # Size of module_exec_dict == 1
            module_name, module_dict = list(module_exec_dict.items())[0]
            # Convert the module_dict into a list of Output
            module_outputs: List[Output] = []

            # Modules that output a single question and paragraph attention
            if module_name in ["find", "filter", "relocate"]:
                question_output = Output(input_name="question", values=module_dict["question"],
                                         label="question_attention")
                passage_output = Output(input_name="passage", values=module_dict["passage"], label="module_output")
                outputs = [question_output, passage_output]
                if "input" in module_dict:
                    passage_input = Output(input_name="passage", values=module_dict["passage_input"], label="module_input")
                    outputs.append(passage_input)
                module_outputs.extend(outputs)

            # Modules that output two date_distributions and one passage distribution
            elif module_name in ["compare-date-lt", "compare-date-gt"]:
                passage_output = Output(input_name="passage", values=module_dict["passage"], label="module_output")
                passage_date_1 = Output(input_name="passage", values=module_dict["passage_date_1"],
                                        label="passage_date_1")
                passage_date_2 = Output(input_name="passage", values=module_dict["passage_date_2"],
                                        label="passage_date_2")
                date_1 = Output(input_name="dates", values=module_dict["date_1"], label="date_1")
                date_2 = Output(input_name="dates", values=module_dict["date_2"], label="date_2")
                module_outputs.extend([passage_output, passage_date_1, passage_date_2, date_1, date_2])

            # Modules that output two dates and a year diff
            elif module_name in ["year-diff"]:
                year_diff = Output(input_name="year_diffs", values=module_dict["year-diff"],
                                   label="output_year_diff_attention")
                passage_date_1 = Output(input_name="passage", values=module_dict["passage_date_1"],
                                        label="passage_date_1")
                passage_date_2 = Output(input_name="passage", values=module_dict["passage_date_2"],
                                        label="passage_date_2")
                date_1 = Output(input_name="dates", values=module_dict["date_1"], label="date_1")
                date_2 = Output(input_name="dates", values=module_dict["date_2"], label="date_2")
                module_outputs.extend([year_diff, passage_date_1, passage_date_2, date_1, date_2])

            # Modules that output two num_distributions and one passage distribution
            elif module_name in ["compare-num-lt", "compare-num-gt"]:
                passage_output = Output(input_name="passage", values=module_dict["passage"], label="module_output")
                passage_number_1 = Output(input_name="passage", values=module_dict["passage_number_1"],
                                        label="passage_number_1")
                passage_number_2 = Output(input_name="passage", values=module_dict["passage_number_2"],
                                        label="passage_number_2")
                number_1 = Output(input_name="numbers", values=module_dict["number_1"], label="number_1")
                number_2 = Output(input_name="numbers", values=module_dict["number_2"], label="number_2")
                module_outputs.extend([passage_output, passage_number_1, passage_number_2, number_1, number_2])

            # Modules that output one num_distribution
            elif module_name in ["find-num"]:
                passage_input = Output(input_name="passage", values=module_dict["passage_input"], label="module_input")
                passage_number = Output(input_name="passage", values=module_dict["passage_number"],
                                        label="passage_number_attention")
                number = Output(input_name="numbers", values=module_dict["number"], label="number_distribution")
                module_outputs.extend([passage_input, passage_number, number])

            # Find-max-num and Find-min-num
            elif module_name in ["find-max-num", "find-min-num"]:
                passage_input = Output(input_name="passage", values=module_dict["passage_input"], label="module_input")
                passage_output = Output(input_name="passage", values=module_dict["passage"], label="module_output")
                input_passage_number = Output(input_name="passage", values=module_dict["passage_input_number"],
                                              label="input_pattn_number_attention")
                minmax_passage_number = Output(input_name="passage", values=module_dict["passage_minmax_number"],
                                               label="minmax_number_attention")
                # Not displaying the input number distribution aggregated over numbers
                # input_number = Output(input_name="numbers", values=module_dict["number_input"],
                #                       label="input_number_distribution")
                module_outputs.extend([passage_input, passage_output, input_passage_number,
                                       minmax_passage_number])

            # Addition subtraction modules
            elif module_name in ["number-difference", "number-addition"]:
                if module_name == "number-difference":
                    output_distribution = module_dict["difference_value"]
                    label = "difference_distribution"
                else:
                    output_distribution = module_dict["addition_value"]
                    label = "addition_distribution"
                composed_number = Output(input_name="composed_numbers", values=output_distribution,
                                         label=label)
                module_outputs.extend([composed_number])

            # Modules that output count
            elif module_name in ["count"]:
                passage_input = Output(input_name="passage", values=module_dict["passage_input"], label="module_input")
                count = Output(input_name="count", values=module_dict["count"], label="module_output")
                module_outputs.extend([passage_input, count])

            # span module
            elif module_name in ["span"]:
                passage_input = Output(input_name="passage", values=module_dict["passage_input"], label="module_input")
                passage_output = Output(input_name="passage", values=module_dict["token_probs"],
                                        label="aggregated_token_probabilities")
                span_probs = Output(input_name="span_probabilities", values=module_dict["span_probs"],
                                    label="span_probabilities")
                module_outputs.extend([passage_input, passage_output, span_probs])

            else:
                continue

            modified_program_execution.append({module_name: module_outputs})

        return modified_program_execution

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
        composed_numbers = metadata["composed_numbers"]
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
        # This contains each module as a dict{"name": module_name, "identifier": stepnum_of_module_exection}
        module_expression, _ = add_identifier(program_nested_expression, count=1)
        program_nested_expression = convert_module_expression_tree_to_dict(module_expression)

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

        # Convert span start/end logits to token probs and span probs
        for module_dict in program_execution:
            for module_name, module_output_dict in module_dict.items():
                if module_name == "span":
                    span_start_logits = module_output_dict["span_start_logits"]
                    span_end_logits = module_output_dict["span_end_logits"]
                    token_probs, span_probs = get_topk_spans(span_start_logits=span_start_logits,
                                                             span_end_logits=span_end_logits, passage_wps=passage_wps,
                                                             passage_tokens=passage_tokens,
                                                             passage_wpidx2tokenidx=passage_wpidx2tokenidx)
                    module_output_dict["token_probs"] = token_probs
                    module_output_dict["span_probs"] = span_probs
                    module_output_dict.pop("span_start_logits")
                    module_output_dict.pop("span_end_logits")


        # Make List[Input] for different kind of inputs that the model gets
        question_Input = Input(name="question", tokens=question_tokens)
        passage_Input = Input(name="passage", tokens=passage_tokens)
        numbers_Input = Input(name="numbers", tokens=num_values)
        dates_Input = Input(name="dates", tokens=date_values)
        year_diffs_Input = Input(name="year_diffs", tokens=year_diff_values)
        composed_numbers_Input = Input(name="composed_numbers", tokens=composed_numbers)
        count_Input = Input(name="count", tokens=list(range(10)))
        inputs: List[Input] = [question_Input, passage_Input, numbers_Input, dates_Input, composed_numbers_Input,
                               year_diffs_Input, count_Input]
        input_jsonserializable = [i.to_dict() for i in inputs]

        # Convert module_outputs in program_execution from Dict to List[Output]
        program_execution = self.convert_module_outputs_to_list(program_execution)
        program_execution_jsonserializable = []
        for module_exec_dict in program_execution:
            # Size of module_exec_dict == 1
            module_name, module_outputs = list(module_exec_dict.items())[0]
            module_outputs_dicts = [o.to_dict() for o in module_outputs]  # module_outputs: List[Output]
            program_execution_jsonserializable.append({module_name: module_outputs_dicts})
        # outputs["program_execution"] = program_execution_jsonserializable

        output_dict = {
            "question": question,
            "passage": passage,
            "predicted_ans": predicted_ans,
            "answer": predicted_ans,
            # "question_tokens": question_tokens,
            # "passage_tokens": passage_tokens,
            # "numbers": num_values,
            # "dates": date_values,
            # "year_diff_values": year_diff_values,
            "inputs": input_jsonserializable,
            "program_nested_expression": program_nested_expression,
            "program_lisp": program_lisp,
            "program_execution": program_execution_jsonserializable,
        }
        return output_dict

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """Convert output from predict_json to JSON serializable due to presence of Input and Output objects."""
        # inputs: List[Input] = outputs["inputs"]
        # input_jsonserializable = [i.to_dict() for i in inputs]
        # outputs["inputs"] = input_jsonserializable
        #
        # program_execution = outputs["program_execution"]
        # program_execution_jsonserializable = []
        # for module_exec_dict in program_execution:
        #     # Size of module_exec_dict == 1
        #     module_name, module_outputs = list(module_exec_dict.items())[0]
        #     module_outputs_dicts = [o.to_dict() for o in module_outputs]   # module_outputs: List[Output]
        #     program_execution_jsonserializable.append({module_name: module_outputs_dicts})
        # outputs["program_execution"] = program_execution_jsonserializable

        return json.dumps(outputs)

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



def get_topk_spans(span_start_logits, span_end_logits, passage_wps, passage_tokens, passage_wpidx2tokenidx):
    max_span_length = 20
    max_num_spans = 100

    span_start_logits = torch.Tensor(span_start_logits).unsqueeze(0)
    span_end_logits = torch.Tensor(span_end_logits).unsqueeze(0)

    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()

    valid_span_log_probs = span_log_probs + span_log_mask

    # Shape: (batch_size, passage_length*passage_length)
    valid_span_log_probs_flat = valid_span_log_probs.view(batch_size, -1)
    valid_span_probs = torch.nn.functional.softmax(valid_span_log_probs_flat, dim=-1)   # computing span probs
    wps_span_probs, span_indices = torch.sort(valid_span_probs, descending=True)
    span_start_indices = span_indices // passage_length     # Span start wordpiece index
    span_end_indices = span_indices % passage_length        # Span end wordpiece index

    # Assuming demo is running with batch_size of 1, hence removed the first dim
    span_start_indices = span_start_indices.detach().cpu().numpy().tolist()[0]
    span_end_indices = span_end_indices.detach().cpu().numpy().tolist()[0]
    wps_span_probs = wps_span_probs.detach().cpu().numpy().tolist()[0]

    # Computing wordpiece-prob as the sum of probability of spans that this wordpiece occurs in
    wps_probs = np.array([0.0] * passage_length)
    wp_span_probs = []      # List of [[start, end], prob]
    for span_num in range(0, max_num_spans):
        start, end = span_start_indices[span_num], span_end_indices[span_num]
        prob = wps_span_probs[span_num]
        wp_span_probs.append([[start, end], prob])
        wps_probs[start:end + 1] += prob   # end + 1 since end is inclusive

    # Converting wordpiece probabilities token-probabilities
    token_probs = convert_wordpiece_attention_to_tokens(wp_attention=wps_probs, tokens=passage_tokens,
                                                        wps=passage_wps, wpidx2tokenidx=passage_wpidx2tokenidx)
    # Converting wps_span_probs list to token-spans
    token_span_probs = []
    for span, prob in wp_span_probs:
        token_start, token_end = passage_wpidx2tokenidx[span[0]], passage_wpidx2tokenidx[span[1]]
        if token_end - token_start > max_span_length:
            continue
        token_span_probs.append([[token_start, token_end], prob])

    return token_probs, token_span_probs
