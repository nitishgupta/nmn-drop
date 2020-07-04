from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import utils.util as myutils


@Predictor.register("pattn2bio_predictor")
class Pattn2CountPredictor(Predictor):
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
        pattn = outputs["passage_attention"]
        pattn = myutils.round_all(pattn, 4)
        gold_spans = outputs["gold_spans"]
        predicted_spans = outputs["predicted_spans"]
        gold_count = outputs["gold_count"]
        predicted_count = outputs["predicted_count"]
        count_distribution = outputs["count_distribution"]
        count_mean = outputs["count_mean"]
        count_correct = predicted_count == gold_count

        pattn_out_str = ""
        for tokenidx, attn in zip(range(len(pattn)), pattn):
            pattn_out_str += f"{tokenidx}|{attn} "
        pattn_out_str = pattn_out_str.strip()

        out_str += f"Passage attention:\n{pattn_out_str}\n"
        out_str += f"Gold Spans: {gold_spans}\n"
        out_str += f"Predicted Spans: {predicted_spans}\n"
        out_str += f"Gold Count: {gold_count}\n"
        out_str += f"Predicted Count: {predicted_count}\n"
        out_str += f"Predicted Count mean: {count_mean}\n"
        out_str += f"Count acc: {count_correct}\n"

        out_str += "--------------------------------------------------\n"


        # psigmoid = outputs["passage_sigmoid"]
        # psigmoid = myutils.round_all(psigmoid, 4)

        # attn_sigm = list(zip(pattn, psigmoid))
        #
        # passage_count_mean = outputs["count_mean"]
        # count_distribution = outputs["count_distritbuion"]
        # count_answer = outputs["count_answer"]
        # pred_count_idx = outputs["pred_count"]
        #
        # out_str += f"Pattn: {pattn}" + "\n"
        # out_str += f"Psigm: {psigmoid}" + "\n"
        # out_str += f"Pattn_sigm: {attn_sigm}" + "\n"
        # out_str += f"Plen: {len(pattn)}" + "\n"
        # out_str += f"PattnSum: {sum(pattn)}" + "\n"
        # out_str += f"PSigmSum: {sum(psigmoid)}" + "\n"
        # out_str += f"CountMean: {passage_count_mean}" + "\n"
        # out_str += f"CountDist: {count_distribution}" + "\n"
        # out_str += f"CountAnswer: {count_answer}" + "\n"
        # out_str += f"Predicted CountAnswer: {pred_count_idx}" + "\n"
        # out_str += "--------------------------------------------------\n"

        return out_str

    # @overrides
    # def _json_to_instance(self, json_dict: JsonDict):
    #     jsonobj = json_dict
    #     # space delimited tokenized
    #     question = jsonobj[hpconstants.q_field]
    #
    #     answer = jsonobj[hpconstants.ans_field]
    #     # List of question mentions. Stored as mention-tuples --- (text, start, end, label)
    #     # TODO(nitish): Fix this to include all types
    #     q_nemens = jsonobj[hpconstants.q_ent_ner_field]
    #     # List of (title, space_delimited_tokenized_contexts)
    #     contexts = jsonobj[hpconstants.context_field]
    #     # List of list --- For each context , list of mention-tuples as (text, start, end, label)
    #     contexts_ent_ners = jsonobj[hpconstants.context_ent_ner_field]
    #     contexts_num_ners = jsonobj[hpconstants.context_num_ner_field]
    #     contexts_date_ners = jsonobj[hpconstants.context_date_ner_field]
    #
    #     # Mention to entity mapping -- used to make the grounding vector
    #     context_entmens2entidx = jsonobj[hpconstants.context_nemens2entidx]
    #     context_nummens2entidx = jsonobj[hpconstants.context_nummens2entidx]
    #     context_datemens2entidx = jsonobj[hpconstants.context_datemens2entidx]
    #
    #     # Entity to mentions --- Used to find the number of entities of each type in the contexts
    #     context_eqent2entmens = jsonobj[hpconstants.context_eqent2entmens]
    #     context_eqent2nummens = jsonobj[hpconstants.context_eqent2nummens]
    #     context_eqent2datemens = jsonobj[hpconstants.context_eqent2datemens]
    #
    #     # Dict from {date_string: (date, month, year)} normalization. -1 indicates invalid field
    #     dates_normalized_dict = jsonobj[hpconstants.dates_normalized_field]
    #     # Dict from {num_string: float_val} normalization.
    #     nums_normalized_dict = jsonobj[hpconstants.nums_normalized_field]
    #     # Dict from {ent_idx: [(context_idx, men_idx)]} --- output pf CDCR
    #
    #     # Grounding of ques entity mentions
    #     qnemens_to_ent = jsonobj[hpconstants.q_entmens2entidx]
    #
    #     ans_type = None
    #     ans_grounding = None
    #     # if hpconstants.ans_type_field in jsonobj:
    #     #     ans_type = jsonobj[hpconstants.ans_type_field]
    #     #     ans_grounding = jsonobj[hpconstants.ans_grounding_field]
    #
    #     instance = self._dataset_reader.text_to_instance(question,
    #                                                      answer,
    #                                                      q_nemens,
    #                                                      contexts,
    #                                                      contexts_ent_ners,
    #                                                      contexts_num_ners,
    #                                                      contexts_date_ners,
    #                                                      context_entmens2entidx,
    #                                                      context_nummens2entidx,
    #                                                      context_datemens2entidx,
    #                                                      context_eqent2entmens,
    #                                                      context_eqent2nummens,
    #                                                      context_eqent2datemens,
    #                                                      dates_normalized_dict,
    #                                                      nums_normalized_dict,
    #                                                      qnemens_to_ent,
    #                                                      ans_type,
    #                                                      ans_grounding)
    #     return instance
