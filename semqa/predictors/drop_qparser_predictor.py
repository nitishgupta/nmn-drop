from typing import List, Union, Dict, Tuple

from dataclasses import dataclass
import json
import pdb
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
from semqa.utils.qdmr_utils import node_from_dict, nested_expression_to_lisp, Node, nested_expression_to_tree, \
    lisp_to_nested_expression, function_to_action_string_alignment
from semqa.utils.prediction_analysis import NMNPredictionInstance

def compute_postorder_position(node: Node, position: int = 0):
    for c in node.children:
        position = compute_postorder_position(c, position)
    node.post_order = position
    position += 1
    return position


def compute_inorder_position(node: Node, position: int = 0):
    node.in_order = position
    position += 1
    for c in node.children:
        position = compute_inorder_position(c, position)
    return position


def compute_postorder_position_in_inorder_traversal(node: Node):
    postorder_positions = [node.post_order]
    for c in node.children:
        c_postorder_positions = compute_postorder_position_in_inorder_traversal(c)
        postorder_positions.extend(c_postorder_positions)
    return postorder_positions


def add_quesattn_to_node(node: Node, inorder_position: int, question_attentions: List[List[float]],
                         question_tokens: List[str], inorder2actionidx: List[int]):
    node.in_order = inorder_position
    actionidx = inorder2actionidx[inorder_position]
    node.extras["predicted_question_attention"] = question_attentions[actionidx]
    node.extras["question_tokens"] = question_tokens

    inorder_position += 1
    for c in node.children:
        inorder_position = add_quesattn_to_node(c, inorder_position, question_attentions,
                                                question_tokens, inorder2actionidx)
    return inorder_position


def nestedexpr_w_predictedquesattn(node: Node, relevant_predicates: List[str], attn_threhold: float):
    node_name = node.predicate
    if node_name in relevant_predicates:
        predicted_ques_attn = node.extras["predicted_question_attention"]
        qtokens = node.extras["question_tokens"]
        string_arg_tokens = [t for t, attn in zip(qtokens, predicted_ques_attn) if attn >= attn_threhold]
        node_name = node_name + "(" + " ".join(string_arg_tokens) + ")"
        node.extras["predicted_ques_string_arg"] = " ".join(string_arg_tokens)

    if not node.children:
        return node_name
    else:
        nested_expression = [node_name]
        for child in node.children:
            nested_expression.append(nestedexpr_w_predictedquesattn(child, relevant_predicates, attn_threhold))
        return nested_expression


class PredictionData:
    def __init__(self, outputs: JsonDict, attn_threshold: float = 0.05):
        self.metadata = outputs["metadata"]
        self.question = self.metadata["question"]
        self.question_id = self.metadata["question_id"]
        self.passage_id = self.metadata["passage_id"]
        self.question_tokens = self.metadata["question_tokens"]
        self.question_wpidx2tokenidx: List[int] = self.metadata["question_wpidx2tokenidx"]

        gold_program_dict = outputs["gold_program_dict"]
        self.gold_program: Node = node_from_dict(gold_program_dict)
        self.top_action_seq = outputs["top_action_seq"]
        self.top_action_prob = outputs["top_action_prob"]
        self.top_action_siderags = outputs["top_action_siderags"]
        self.top_lisp = outputs["top_action_lisp"]

        self.top_nested_expr = []
        self.top_program_node = None
        self.top_program_dict = {}
        self.program_question_attentions = []
        self.nestedexp_w_predictedstrings = []

        if not self.top_action_seq:
            return

        self.top_nested_expr = lisp_to_nested_expression(self.top_lisp)
        self.top_program_node: Node = nested_expression_to_tree(self.top_nested_expr)
        self.top_program_dict = self.top_program_node.to_dict()

        # Question-wordpiece-attention for all actions
        self.program_question_attentions: List[List[float]] = [sidearg["question_attention"]
                                                               for sidearg in self.top_action_siderags]

        # Converting word-piece attention to token-attention
        program_token_attentions = []
        for actionidx, action_question_attn in enumerate(self.program_question_attentions):
            token_attentions = [0.0 for _ in self.question_tokens]
            for wpidx, tokenidx in enumerate(self.question_wpidx2tokenidx):
                if 0 <= tokenidx < len(self.question_tokens):
                    if 0 <= wpidx < len(action_question_attn):
                        token_attentions[tokenidx] += action_question_attn[wpidx]
                    # else:
                    #     print(f"wpidx: {wpidx} is out of bounds. Len-ques-attn: {len(action_question_attn)}")
                # else:
                #     print(f"tokenidx: {tokenidx} is out of bounds. Len-ques-tokens: {len(question_tokens)}")
            program_token_attentions.append(token_attentions)

        # for program-functions listed in inorder, mapping to the equivalent action-idx; this can be used, for example,
        # to get the prediction question-attention when decoding these functions
        inorder_functionidx2actionidx: List[int] = function_to_action_string_alignment(
            program_node=self.top_program_node, action_strings=self.top_action_seq)

        # Add the prediction question-attention to each node (even if the node-function need one for execution
        # This is added in a new member, predicted_ques_attn, in the Node
        add_quesattn_to_node(node=self.top_program_node,
                             inorder_position=0,
                             question_attentions=program_token_attentions,
                             question_tokens=self.question_tokens,
                             inorder2actionidx=inorder_functionidx2actionidx)

        # All functions that take a question-string argument
        functions_w_string_args = ["select_passage", "filter_passage", "project_passage", "select_implicit_num"]

        # nested expression with predicted-string-arg for relevant functions; all tokens with attn >= threshold
        self.nestedexp_w_predictedstrings = nestedexpr_w_predictedquesattn(
            node=self.top_program_node, relevant_predicates=functions_w_string_args, attn_threhold=attn_threshold)


@Predictor.register("drop_qparser_predictor")
class DropQuesParserPredictor(Predictor):
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
        prediction_data = PredictionData(outputs, attn_threshold=0.05)

        out_str = ""

        out_str += "qid: {} \t pid: {}".format(prediction_data.question_id, prediction_data.passage_id) + "\n"
        out_str += prediction_data.question + "\n"

        out_str += "Gold-program: {}\n".format(prediction_data.gold_program.get_nested_expression_with_strings())
        out_str += "Predicted-program-prob: {}\n".format(prediction_data.top_action_prob)
        out_str += "Pred-program: {}\n".format(prediction_data.nestedexp_w_predictedstrings)

        out_str += "--------------------------------------------------\n\n"

        return out_str


@Predictor.register("drop_qparser_jsonl_predictor")
class DROPQParserJSONLPredictor(DropQuesParserPredictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        # Use json.dumps(outputs) + "\n" to dump a dictionary

        prediction_data = PredictionData(outputs=outputs, attn_threshold=0.05)

        output_dict = {
            "question": prediction_data.question,
            "query_id": prediction_data.question_id,
            "gold_logical_form": nested_expression_to_lisp(prediction_data.gold_program.get_nested_expression()),
            "gold_nested_expr": prediction_data.gold_program.get_nested_expression(),
            "gold_nested_expr_wstr": prediction_data.gold_program.get_nested_expression_with_strings(),
            "top_logical_form": prediction_data.top_lisp,
            "top_nested_expr_wstr": prediction_data.nestedexp_w_predictedstrings,
            "top_logical_form_prob": prediction_data.top_action_prob,
            "top_program_dict": prediction_data.top_program_dict,
        }

        return json.dumps(output_dict) + "\n"
