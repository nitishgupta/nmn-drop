from typing import List, Union, Dict, Tuple

from dataclasses import dataclass
import json
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp_semparse.common.util import lisp_to_nested_expression
from semqa.domain_languages.domain_language_utils import mostAttendedSpans, listTokensVis
from semqa.domain_languages.drop_language import Output, output_from_dict

from semqa.utils.squad_eval import metric_max_over_ground_truths
from semqa.utils.drop_eval import get_metrics as drop_em_and_f1, answer_json_to_strings
from semqa.utils.qdmr_utils import node_from_dict, nested_expression_to_lisp, Node, nested_expression_to_tree
from semqa.utils.prediction_analysis import NMNPredictionInstance

def compute_postorder_position(node: Node, position: int = 0):
    for c in node.children:
        position = compute_postorder_position(c, position)
    node.post_order = position
    position += 1
    return position


def compute_postorder_position_in_inorder_traversal(node: Node):
    postorder_positions = [node.post_order]
    for c in node.children:
        c_postorder_positions = compute_postorder_position_in_inorder_traversal(c)
        postorder_positions.extend(c_postorder_positions)
    return postorder_positions


class PredictionData:
    def __init__(self, outputs: JsonDict):
        self.metadata: Dict = outputs["metadata"]
        self.predicted_ans: str = outputs["predicted_answer"]

        self.question_mask: List[float] = outputs["question_mask"]
        self.passage_mask: List[float] = outputs["passage_mask"]
        self.logical_forms: List[str] = outputs["logical_forms"]
        self.top_logical_form: str = self.logical_forms[0] if self.logical_forms else ""
        self.top_nested_expr = []
        if self.top_logical_form:
            self.top_nested_expr: List = lisp_to_nested_expression(self.top_logical_form)
        self.actionseq_logprobs: List[float] = outputs["actionseq_logprobs"]
        self.actionseq_probs: List[float] = outputs["actionseq_probs"]
        self.top_logical_form_prob: float = self.actionseq_probs[0] if self.actionseq_probs else -1
        self.prog_denotations: List[str] = outputs["all_predicted_answers"]

        modules_debug_infos = outputs["modules_debug_infos"]
        program_execution: List[Dict] = modules_debug_infos[0] if modules_debug_infos else []
        if self.top_nested_expr:
            program_node: Node = nested_expression_to_tree(self.top_nested_expr)
            compute_postorder_position(program_node)   # adding positions to nodes if traversed in post-order
            # These are the post-order positions of nodes when tree is traversed in-order manner
            postorder_position_in_inorder_traversal = compute_postorder_position_in_inorder_traversal(program_node)
            if len(postorder_position_in_inorder_traversal) == len(program_execution):
                program_execution_vis = [program_execution[x] for x in postorder_position_in_inorder_traversal]
            else:
                program_execution_vis = program_execution[::-1]
                print("\nLen of program : {}".format(len(postorder_position_in_inorder_traversal)))
                print(f"Program: {self.top_nested_expr}")
                print(f"Program execution: {[key for key in program_execution]}")
                print()
        else:
            # Some thing is wrong before; resort to assuming that the tree is fully left-branching
            program_execution_vis = program_execution[::-1]

        # Each dict is a singleton {module_name: List[Output-dict]} where the Output is a serialized as a Dict
        self.program_execution: List[Dict] = program_execution_vis

        self.gold_passage_span_ans: List[Tuple[int, int]] = self.metadata["answer_passage_spans"] if \
            "answer_passage_spans" in self.metadata else []
        self.gold_question_span_ans: List[Tuple[int, int]] = self.metadata["answer_question_spans"] if \
            "answer_question_spans" in self.metadata else []

        self.question_id: str = self.metadata["question_id"]
        self.passage_id: str = self.metadata["passage_id"]
        self.question: str = self.metadata["question"]
        self.passage: str = self.metadata["passage"]
        self.unpadded_q_wps_len: int = self.metadata["unpadded_q_wps_len"]
        self.question_wps: List[str] = self.metadata["question_wps"][0:self.unpadded_q_wps_len]
        self.passage_tokens: List[str] = self.metadata["passage_tokens"]
        self.passage_wps: List[str] = self.metadata["passage_wps"]
        self.passage_wpidx2tokenidx: List[int] = self.metadata["passage_wpidx2tokenidx"]
        self.question_wpidx2tokenidx: List[int] = self.metadata["question_wpidx2tokenidx"]
        self.answer_annotation_dicts: List[Dict] = self.metadata["answer_annotations"]
        self.passage_date_values: List[str] = self.metadata["passage_date_values"]
        self.passage_number_values: List[float] = self.metadata["passage_number_values"]
        self.composed_numbers: List[float] = self.metadata["composed_numbers"]
        self.passage_year_diffs: List[int] = self.metadata["passage_year_diffs"]
        self.count_values: List[int] = self.metadata["count_values"]
        self.program_supervision: Union[None, Dict] = self.metadata.get("program_supervision", None)
        if self.program_supervision == "None":      # sanitize converts NoneType to "None"
            self.program_supervision = None

        (self.exact_match, self.f1_score) = f1metric(self.predicted_ans, self.answer_annotation_dicts)


def visualize_module_outputs(module_name: str, module_outputs: List[Dict], prediction_data: PredictionData):
    """For a module and its list of Output(s) return a string-output that can be visualized."""
    outputs: List[Output] = [output_from_dict(x) for x in module_outputs]
    output_str = f"{module_name}\n"
    for output in outputs:
        output_str += visualize_single_output_from_module(output, prediction_data) + '\n'
    output_str = output_str.strip() + "\n\n"
    return output_str


def visualize_single_output_from_module(output: Output, prediction_data: PredictionData):
    """Visualize string for single Output attention from a module. """
    output_values = {'passage': prediction_data.passage_wps,
                     'question': prediction_data.question_wps,
                     'numbers': prediction_data.passage_number_values,
                     'dates': prediction_data.passage_date_values,
                     'year_diffs': prediction_data.passage_year_diffs,
                     'composed_numbers': prediction_data.composed_numbers,
                     'count': prediction_data.count_values}
    output_type: str = output.output_type
    output_support: List[Union[str, int, float]] = output_values[output_type]
    output_attn: List[float] = output.values
    output_label: str = output.label

    output_str = f"\t{output_label}\n"
    complete_attention_vis, _ = listTokensVis(attention_vec=output_attn, tokens=output_support)
    output_str += "\t" + complete_attention_vis + "\n"
    if output_type in ["passage"]:
        most_attended_spans = mostAttendedSpans(attention_vec=output_attn, tokens=output_support)
        output_str += "\t" + most_attended_spans + "\n"

    return output_str


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


@Predictor.register("drop_parser_predictor")
class DropNMNPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

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

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        # Use json.dumps(outputs) + "\n" to dump a dictionary

        prediction_data = PredictionData(outputs=outputs)
        program_node = prediction_data.program_supervision
        if program_node is None:
            gold_program = "N/A"
        else:
            program_node = node_from_dict(program_node)
            gold_program = program_node.get_nested_expression_with_strings()

        out_str = ""
        out_str += "qid: {} \t pid: {}".format(prediction_data.question_id, prediction_data.passage_id) + "\n"
        out_str += prediction_data.question + "\n"
        out_str += prediction_data.passage + "\n"

        out_str += f"GoldAnswer: {prediction_data.answer_annotation_dicts}" + "\n"
        out_str += f"GoldPassageSpans:{prediction_data.gold_passage_span_ans}  " \
                   f"GoldQuesSpans:{prediction_data.gold_question_span_ans}\n"
        out_str += f"GoldProgram: {gold_program}\n\n"

        out_str += f"PredictedAnswer: {prediction_data.predicted_ans}" + "\n"
        out_str += f"F1:{prediction_data.f1_score} EM:{prediction_data.exact_match}" + "\n"

        out_str += f"Top-logical-form: {prediction_data.top_logical_form}" + "\n"
        out_str += f"Top-prog-Prob: {prediction_data.top_logical_form_prob}" + "\n"

        out_str += f"Dates: {prediction_data.passage_date_values}" + "\n"
        out_str += f"PassageNums: {prediction_data.passage_number_values}" + "\n"
        out_str += f"ComposedNumbers: {prediction_data.composed_numbers}" + "\n"
        out_str += f"YearDiffs: {prediction_data.passage_year_diffs}" + "\n"

        out_str += "Program execution" + "\n"
        for module_output_dict in prediction_data.program_execution:
            for module_name, module_outputs in module_output_dict.items():
                module_output_str = visualize_module_outputs(module_name=module_name, module_outputs=module_outputs,
                                                             prediction_data=prediction_data)
                out_str += module_output_str

        out_str += "--------------------------------------------------\n"

        return out_str


@Predictor.register("drop_parser_jsonl_predictor")
class DROPNMNJSONLPredictor(DropNMNPredictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        # Use json.dumps(outputs) + "\n" to dump a dictionary

        prediction_data = PredictionData(outputs=outputs)

        # Node in a dict format
        program_supervision: Union[Dict, None] = prediction_data.program_supervision
        gold_program_lisp = ""
        gold_nested_expr = []
        if program_supervision:
            program_node = node_from_dict(program_supervision)
            nested_expr = program_node.get_nested_expression()
            gold_program_lisp = nested_expression_to_lisp(nested_expr)
            gold_nested_expr = program_node.get_nested_expression_with_strings()

        output_dict = {
            "question": prediction_data.question,
            "query_id": prediction_data.question_id,
            "gold_logical_form": gold_program_lisp,
            "gold_nested_expr": gold_nested_expr,
            "predicted_ans": prediction_data.predicted_ans,
            "top_logical_form": prediction_data.top_logical_form,
            "top_nested_expr": prediction_data.top_nested_expr,
            "top_logical_form_prob": prediction_data.top_logical_form_prob,
            "f1": prediction_data.f1_score,
            "em": prediction_data.exact_match
        }

        return json.dumps(output_dict) + "\n"


@Predictor.register("drop_analysis_predictor")
class DropQANetPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        # Use json.dumps(outputs) + "\n" to dump a dictionary

        out_str = ""
        metadata = outputs["metadata"]
        predicted_ans = outputs["predicted_answer"]

        # instance_spans_for_all_progs = outputs['predicted_spans']
        # best_span = instance_spans_for_all_progs[0]
        question_id = metadata["question_id"]
        question = metadata["original_question"]
        answer_annotation_dicts = metadata["answer_annotations"]
        (exact_match, f1_score) = f1metric(predicted_ans, answer_annotation_dicts)

        correct_or_not = "NC"
        if f1_score >= 0.75:
            correct_or_not = "C"

        logical_form = outputs["logical_forms"][0]

        out_str += question_id + "\t"
        out_str += question + "\t"
        out_str += correct_or_not + "\t"
        out_str += logical_form + "\n"

        return out_str


@Predictor.register("drop_mtmsnstyle_predictor")
class MTMSNStylePredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        # Use json.dumps(outputs) + "\n" to dump a dictionary

        out_str = ""
        metadata = outputs["metadata"]
        predicted_ans = outputs["predicted_answer"]

        # instance_spans_for_all_progs = outputs['predicted_spans']
        # best_span = instance_spans_for_all_progs[0]
        question_id = metadata["question_id"]
        question = metadata["original_question"]
        answer_annotation_dicts = metadata["answer_annotations"]
        program_probs = outputs["batch_actionseq_probs"]  # List size is the same as number of programs predicted
        (exact_match, f1_score) = f1metric(predicted_ans, answer_annotation_dicts)
        logical_forms = outputs["logical_forms"]
        if logical_forms:
            logical_form = logical_forms[0]
        else:
            logical_form = "NO PROGRAM PREDICTED"
        if program_probs:
            prog_prob = program_probs[0]
        else:
            prog_prob = 0.0

        output_dict = {
            "query_id": question_id,
            "question": question,
            "text": predicted_ans,
            "type": logical_form,
            "prog_prob": prog_prob,
            "f1": f1_score,
            "em": exact_match,
            "gold_answer": answer_annotation_dicts
        }

        return json.dumps(output_dict) + "\n"


@Predictor.register("drop_interpret_predictor")
class DropNMNPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

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

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        # Use json.dumps(outputs) + "\n" to dump a dictionary

        metadata = outputs["metadata"]
        predicted_ans = outputs["predicted_answer"]
        module_debug_infos = outputs["modules_debug_infos"]
        passage_mask = outputs["passage_mask"]
        passage_token_idxs = outputs["passage_token_idxs"]

        gold_passage_span_ans = metadata["answer_passage_spans"] if "answer_passage_spans" in metadata else []
        gold_question_span_ans = metadata["answer_question_spans"] if "answer_question_spans" in metadata else []

        # instance_spans_for_all_progs = outputs['predicted_spans']
        # best_span = instance_spans_for_all_progs[0]
        question_id = metadata["question_id"]
        question = metadata["original_question"]
        qtype = metadata["qtype"]
        passage = metadata["original_passage"]
        passage_id = metadata["passage_id"]
        passage_tokens = metadata["passage_orig_tokens"]
        passage_wps = metadata["passage_tokens"]
        passage_wpidx2tokenidx = metadata["passage_wpidx2tokenidx"]
        answer_annotation_dicts = metadata["answer_annotations"]
        passage_date_values = metadata["passage_date_values"]
        passage_num_values = metadata["passage_number_values"]
        composed_numbers = metadata["composed_numbers"]
        passage_year_diffs = metadata["passage_year_diffs"]
        (exact_match, f1_score) = f1metric(predicted_ans, answer_annotation_dicts)

        output_dict = {
            "passage_id": passage_id,
            "query_id": question_id,
            "question": question,
            "qtype": qtype,
            "f1": f1_score,
            "em": exact_match
        }
        logical_forms = outputs["logical_forms"]
        best_logical_form = logical_forms[0]
        output_dict["predicted_logical_form"] = best_logical_form

        # List of dictionary where each dictionary contains a single module_name: pattn-value pair
        module_debug_info: List[Dict] = module_debug_infos[0]
        # This is the length of the passage-wps w/o [SEP] at the end
        len_passage_wps = int(sum(passage_mask)) - 1
        output_dict["module_outputs"] = []
        module_names = ""
        for module_dict in module_debug_info:
            passage_attention = [0.0] * len(passage_tokens)
            module_name, pattn_wps = list(module_dict.items())[0]
            module_names += module_name + " "
            pattn_wps = pattn_wps[:len_passage_wps]  # passage-attention over word-pieces
            for token_idx, attn_value in zip(passage_wpidx2tokenidx, pattn_wps):
                passage_attention[token_idx] += attn_value
            output_dict["module_outputs"].append((module_name, passage_attention))

        return json.dumps(output_dict) + "\n"