from typing import List, Union

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import datasets.hotpotqa.utils.constants as hpconstants
import utils.util as myutils

from allennlp.tools.squad_eval import metric_max_over_ground_truths
from allennlp.tools.drop_eval import (get_metrics as drop_em_and_f1,
                                      answer_json_to_strings)

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
    exact_match, f1_score = metric_max_over_ground_truths(
        drop_em_and_f1,
        prediction,
        ground_truth_answer_strings
    )

    return (exact_match, f1_score)


@Predictor.register("drop_qanet_predictor")
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

        out_str = ''
        metadata = outputs['metadata']
        predicted_ans = outputs['best_span_str']

        question = metadata['original_question']
        passage = metadata['original_passage']
        answer_annotation_dict = metadata['answer_annotation']
        (exact_match, f1_score) = f1metric(predicted_ans, [answer_annotation_dict])

        out_str += question + '\n'
        out_str += passage + '\n'
        out_str += f'GoldAnswer: {answer_annotation_dict}' + '\n'
        out_str += f'PredictedAnswer: {predicted_ans}' + '\n'
        out_str += f'F1:{f1_score} EM:{exact_match}' + '\n'

        out_str += '\n'

        return out_str

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        jsonobj = json_dict
        # space delimited tokenized
        question = jsonobj[hpconstants.q_field]

        answer = jsonobj[hpconstants.ans_field]
        # List of question mentions. Stored as mention-tuples --- (text, start, end, label)
        # TODO(nitish): Fix this to include all types
        q_nemens = jsonobj[hpconstants.q_ent_ner_field]
        # List of (title, space_delimited_tokenized_contexts)
        contexts = jsonobj[hpconstants.context_field]
        # List of list --- For each context , list of mention-tuples as (text, start, end, label)
        contexts_ent_ners = jsonobj[hpconstants.context_ent_ner_field]
        contexts_num_ners = jsonobj[hpconstants.context_num_ner_field]
        contexts_date_ners = jsonobj[hpconstants.context_date_ner_field]

        # Mention to entity mapping -- used to make the grounding vector
        context_entmens2entidx = jsonobj[hpconstants.context_nemens2entidx]
        context_nummens2entidx = jsonobj[hpconstants.context_nummens2entidx]
        context_datemens2entidx = jsonobj[hpconstants.context_datemens2entidx]

        # Entity to mentions --- Used to find the number of entities of each type in the contexts
        context_eqent2entmens = jsonobj[hpconstants.context_eqent2entmens]
        context_eqent2nummens = jsonobj[hpconstants.context_eqent2nummens]
        context_eqent2datemens = jsonobj[hpconstants.context_eqent2datemens]

        # Dict from {date_string: (date, month, year)} normalization. -1 indicates invalid field
        dates_normalized_dict = jsonobj[hpconstants.dates_normalized_field]
        # Dict from {num_string: float_val} normalization.
        nums_normalized_dict = jsonobj[hpconstants.nums_normalized_field]
        # Dict from {ent_idx: [(context_idx, men_idx)]} --- output pf CDCR

        # Grounding of ques entity mentions
        qnemens_to_ent = jsonobj[hpconstants.q_entmens2entidx]

        ans_type = None
        ans_grounding = None
        # if hpconstants.ans_type_field in jsonobj:
        #     ans_type = jsonobj[hpconstants.ans_type_field]
        #     ans_grounding = jsonobj[hpconstants.ans_grounding_field]

        instance = self._dataset_reader.text_to_instance(question,
                                                         answer,
                                                         q_nemens,
                                                         contexts,
                                                         contexts_ent_ners,
                                                         contexts_num_ners,
                                                         contexts_date_ners,
                                                         context_entmens2entidx,
                                                         context_nummens2entidx,
                                                         context_datemens2entidx,
                                                         context_eqent2entmens,
                                                         context_eqent2nummens,
                                                         context_eqent2datemens,
                                                         dates_normalized_dict,
                                                         nums_normalized_dict,
                                                         qnemens_to_ent,
                                                         ans_type,
                                                         ans_grounding)
        return instance
