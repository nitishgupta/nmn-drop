from typing import List, Union, Dict

import json
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import utils.util as myutils

from semqa.utils.squad_eval import metric_max_over_ground_truths
from semqa.utils.drop_eval import get_metrics as drop_em_and_f1, answer_json_to_strings


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

        out_str = ""
        metadata = outputs["metadata"]
        predicted_ans = outputs["predicted_answer"]
        module_debug_infos = outputs["modules_debug_infos"]

        gold_passage_span_ans = metadata["answer_passage_spans"] if "answer_passage_spans" in metadata else []
        gold_question_span_ans = metadata["answer_question_spans"] if "answer_question_spans" in metadata else []

        # instance_spans_for_all_progs = outputs['predicted_spans']
        # best_span = instance_spans_for_all_progs[0]
        question_id = metadata["question_id"]
        question = metadata["original_question"]
        passage = metadata["original_passage"]
        passage_tokens = metadata["passage_orig_tokens"]
        passage_wps = metadata["passage_tokens"]
        passage_wpidx2tokenidx = metadata["passage_wpidx2tokenidx"]
        answer_annotation_dicts = metadata["answer_annotations"]
        passage_date_values = metadata["passage_date_values"]
        passage_num_values = metadata["passage_number_values"]
        composed_numbers = metadata["composed_numbers"]
        passage_year_diffs = metadata["passage_year_diffs"]
        # passage_num_diffs = metadata['passagenum_diffs']
        (exact_match, f1_score) = f1metric(predicted_ans, answer_annotation_dicts)

        out_str += "qid: {}".format(question_id) + "\n"
        out_str += question + "\n"
        out_str += passage + "\n"

        out_str += f"GoldAnswer: {answer_annotation_dicts}" + "\n"
        out_str += f"GoldPassageSpans:{gold_passage_span_ans}  GoldQuesSpans:{gold_question_span_ans}\n"
        # out_str += f"GoldPassageSpans:{answer_as_passage_spans}" + '\n'

        # out_str += f"PredPassageSpan: {best_span}" + '\n'
        out_str += f"PredictedAnswer: {predicted_ans}" + "\n"
        out_str += f"F1:{f1_score} EM:{exact_match}" + "\n"
        if outputs['logical_forms']:
            out_str += f"Top-Prog: {outputs['logical_forms'][0]}" + "\n"
        else:
            out_str += f"Top-Prog: NO PROGRAM FOUND" + "\n"
        if outputs['batch_actionseq_probs']:
            out_str += f"Top-Prog-Prob: {outputs['batch_actionseq_probs'][0]}" + "\n"
        else:
            out_str += f"Top-Prog-Prob: NO PROGRAM FOUND" + "\n"
        out_str += f"Dates: {passage_date_values}" + "\n"
        out_str += f"PassageNums: {passage_num_values}" + "\n"
        out_str += f"ComposedNumbers: {composed_numbers}" + "\n"
        # out_str += f'PassageNumDiffs: {passage_num_diffs}' + '\n'
        out_str += f"YearDiffs: {passage_year_diffs}" + "\n"

        logical_forms = outputs["logical_forms"]
        program_probs = outputs["batch_actionseq_probs"]
        execution_vals = outputs["execution_vals"]
        program_logprobs = outputs["batch_actionseq_logprobs"]
        all_predicted_answers = outputs["all_predicted_answers"]
        if "logical_forms":
            for lf, d, ex_vals, prog_logprob, prog_prob in zip(
                logical_forms, all_predicted_answers, execution_vals, program_logprobs, program_probs
            ):
                ex_vals = myutils.round_all(ex_vals, 1)
                # Stripping the trailing new line
                ex_vals_str = self._print_ExecutionValTree(ex_vals, 0).strip()
                out_str += f"LogicalForm: {lf}\n"
                out_str += f"Prog_LogProb: {prog_logprob}\n"
                out_str += f"Prog_Prob: {prog_prob}\n"
                out_str += f"Answer: {d}\n"
                out_str += f"ExecutionTree:\n{ex_vals_str}"
                out_str += f"\n"
                # NUM_PROGS_TO_PRINT -= 1
                # if NUM_PROGS_TO_PRINT == 0:
                #     break

        # # This is the top scoring program
        # # List of dictionary where each dictionary contains a single module_name: pattn-value pair
        # module_debug_info: List[Dict] = module_debug_infos[0]
        # for module_dict in module_debug_info:
        #     module_name, pattn = list(module_dict.items())[0]
        #     print(module_name)
        #     print(f"{len(pattn)}  {len(passage_wpidx2tokenidx)}")
        #     assert len(pattn) == len(passage_wpidx2tokenidx)
        #
        # # print(module_debug_infos)
        # # print(passage_wpidx2tokenidx)

        out_str += "--------------------------------------------------\n"

        return out_str


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