import sys
import logging
from typing import List, Dict, Any, Tuple

from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions import LinkingTransitionFunction
from allennlp.modules.span_extractors import SpanExtractor
import allennlp.nn.util as allenutil
import allennlp.common.util as alcommon_util
from allennlp.models.archival import load_archive
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow

import semqa.type_declarations.semqa_type_declaration_wques as types
from semqa.domain_languages.hotpotqa.hotpotqa_language_w_sideargs import HotpotQALanguageWSideArgs, Qent, Qstr
from semqa.models.hotpotqa.hotpotqa_parser_base import HotpotQAParserBase
from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters

from semqa.data.datatypes import DateField, NumberField
from semqa.state_machines.constrained_beam_search import ConstrainedBeamSearch
from allennlp.training.metrics import Average

import datasets.hotpotqa.utils.constants as hpcons

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction

@Model.register("bidaf_model")
class BidafModel(Model):
    """
    """

    def __init__(self,
                 vocab: Vocabulary,
                 bidaf_model_path: str,
                 bidaf_wordemb_file: str,
                 bool_bilinear: SimilarityFunction = None,
                 dropout: float = 0.0) -> None:
        super(BidafModel, self).__init__(vocab=vocab)

        self._bool_bilinear = bool_bilinear
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        if bidaf_model_path is None:
            logger.info(f"NOT loading pretrained bidaf model. bidaf_model_path - not given")
            raise NotImplementedError
        logger.info(f"Loading BIDAF model from {bidaf_model_path}")
        bidaf_model_archive = load_archive(bidaf_model_path)
        self.bidaf_model: BidirectionalAttentionFlow = bidaf_model_archive.model

        print('Printing parameter names')

        logger.info(f"Bidaf model successfully loaded!")
        logger.info(f"Extending bidaf model's embedders based on the extended_vocab")
        logger.info(f"Preatrained word embedding file being used: {bidaf_wordemb_file}")

        for key, _ in self.bidaf_model._text_field_embedder._token_embedders.items():
            token_embedder = getattr(self.bidaf_model._text_field_embedder, 'token_embedder_{}'.format(key))
            if isinstance(token_embedder, Embedding):
                token_embedder.extend_vocab(extended_vocab=vocab, pretrained_file=bidaf_wordemb_file)
        logger.info(f"Embedder for bidaf extended. New size: {token_embedder.weight.size()}")
        self.bidaf_encoder_bidirectional = self.bidaf_model._phrase_layer.is_bidirectional()

        self.average_metric = Average()


    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                # quesspans2idx: List[Dict],
                # quesspans_spans: torch.LongTensor,
                # q_nemens_grounding: torch.FloatTensor,
                # q_nemenspan2entidx: List[Dict],
                contexts: Dict[str, torch.LongTensor],
                ent_mens: torch.LongTensor,
                num_nents: torch.LongTensor,
                num_mens: torch.LongTensor,
                num_numents: torch.LongTensor,
                date_mens: torch.LongTensor,
                num_dateents: torch.LongTensor,
                num_normval: List[List[NumberField]],
                date_normval: List[List[DateField]],
                # languages: List[HotpotQALanguageWSideArgs],
                # actions: List[List[ProductionRule]],
                # linked_rule2idx: List[Dict],
                # action2ques_linkingscore: torch.FloatTensor,
                # ques_str_action_spans: torch.LongTensor,
                gold_ans_type: List[str]=None,
                **kwargs) -> Dict[str, torch.Tensor]:

        """
        Forward call to the parser.

        Parameters:
        -----------
        question : Dict[str, torch.LongTensor]
            Question from TextField
        q_qstr2idx : List[Dict]
            For each q, Dict from QSTR to idx into the qstr_span_field (q_qstr_spans)
        q_qstr_spans : torch.LongTensor,
            (B, Q_STR, 2) Spanfield for each QSTR in the q. Padding is -1
        q_nemenspan2entidx: List[Dict],
            Map from q_ne_ent span to entity grounding idx.
            This can be used to directly index into the context mentions tensor
        contexts: Dict[str, torch.LongTensor]
            TextField for each context wrapped into a ListField. Appropriate squeeze/unsqueeze needed to work with this
        ent_mens : ``torch.LongTensor``
            (B, E, C, M, 2) -- for each instance, for each entity, for each context, it's mentions from SpanField
        num_nents : ``torch.LongTensor``
            [B] sized tensor of number of NE entities in the instance
        num_mens : torch.LongTensor
            (B, E, C, M, 2) -- for each instance, for each entity, for each context, it's mentions from SpanField
        num_numents: torch.LongTensor
            [B] sized tensor of number of NUM entities in the instance
        date_mens : torch.LongTensor
            (B, E, C, M, 2) -- for each instance, for each entity, for each context, it's mentions from SpanField
        num_dateents : torch.LongTensor
            [B] sized tensor of number of DATE entities in the instance
        num_normval : List[List[NumberField]]
            For each instance, for each NUM entity it's normalized value
        date_normval: List[List[DateField]]
            For each instance, for each DATE entity it's normalized value
        languages: List[HotpotQALanguage]
            DomainLanguage object for each instance
        actions: List[List[ProductionRule]]
            For each instance, List of all possible ProductionRules (global and instance-specific)
        linked_rule2idx: List[Dict[str, int]]
            Dict from instance-specific-actions to idx into linking_score matrix (action2ques_linkingscore) and
            SpanField indices (ques_str_action_spans)
            Currently this is possible since both instance-specific-action types (QSTR & QENT)
        action2ques_linkingscore: torch.FloatTensor
            (B, A, Q_len): Foe each instance, for each instance-specific action, the linking score for LinkingTF.
            For each action (ques_span for us) a binary vector indicating presence of q_token in A span
        ques_str_action_spans: torch.LongTensor
            (B, A, 2): For each instance_specific action (ques span for us) it's span in Q to make action embeddings
        gold_ans_type: List[str]=None
            List of Ans's types from datasets.hotpotqa.constants
        kwargs: ``Dict``
            Is a dictionary containing datatypes for answer_groundings of different types, key being ans_grounding_TYPE
            Each key has a batched_tensor of ground_truth for the instances.
            Since each instance only has a single correct type, the values for the incorrect types are empty vectors ..
            to make batching easier. See the reader code for more details.
        """
        # pylint: disable=arguments-differ

        batch_size = ent_mens.size()[0]
        if 'metadata' in kwargs:
            metadata = kwargs['metadata']
        else:
            metadata = None

        (ans_grounding_dict,
         ans_grounding_mask) = self._goldAnsDict_and_Masks(gold_ans_type, **kwargs)


        # Finding the number of contexts in the given batch
        (tokenindexer, indexed_tensor)  = next(iter(contexts.items()))
        num_contexts = indexed_tensor.size()[1]

        # To pass each context separately to bidaf
        # Making a separate batched token_indexer_dict for each context -- [{token_inderxer: (B, T, *)}]
        contexts_indices_list: List[Dict[str, torch.LongTensor]] = [{} for _ in range(num_contexts)]
        for token_indexer_name, token_indices_tensor in contexts.items():
            # print(f"{token_indexer_name}  : {token_indices_tensor.size()}")
            for i in range(num_contexts):
                # For a tensor shape (B, C, *), this will slice from dim-1 a tensor of shape (B, *)
                contexts_indices_list[i][token_indexer_name] = token_indices_tensor[:, i, ...]

        # For each context, a repr for question and context from the bidaf model
        bidaf_ques_embed = []           # Just embedded token repr for the question
        bidaf_ques_reprs = []
        bidaf_ques_masks = []
        bidaf_context_reprs = []
        bidaf_context_vecs = []
        bidaf_context_masks = []

        for context in contexts_indices_list:
            bidaf_output_dict = self.bidaf_model(question=question, passage=context)
            bidaf_ques_embed.append(bidaf_output_dict['embedded_question'])
            bidaf_ques_reprs.append(bidaf_output_dict['encoded_question'])
            bidaf_ques_masks.append(bidaf_output_dict['question_mask'])
            bidaf_context_reprs.append(bidaf_output_dict['modeled_passage'])
            bidaf_context_vecs.append(bidaf_output_dict['passage_vector'])
            bidaf_context_masks.append(bidaf_output_dict['passage_mask'])


        # Shape: (B, Qlen, D) -- D is the embedder dimension, but in bidaf is same as encoded repr.
        ques_embed = bidaf_ques_embed[0]
        # Since currently we're using the output of first LSTM for ques_repr, differnt context runs will result
        # in same repr. as there is no interaction between the context and question at this stage.
        # Later, self._average_ques_repr function can be used to average reprs from different runs
        # Shape (B, Qlen, D)
        ques_repr = bidaf_ques_reprs[0]
        ques_mask = bidaf_ques_masks[0]     # Shape: (B, T)
        # List of (Qlen, D) x 2 and (Qlen) resp.
        ques_embed_list = [ques_embed[i] for i in range(batch_size)]
        ques_repr_list = [ques_repr[i] for i in range(batch_size)]
        ques_mask_list = [ques_mask[i] for i in range(batch_size)]
        # Shape: (B, D)
        ques_encoded_final_state = allenutil.get_final_encoder_states(ques_repr,
                                                                      ques_mask,
                                                                      self.bidaf_encoder_bidirectional)

        # List of (C, T, D), (C, D) and (T, D) resp.
        context_repr_list, context_vec_list, context_mask_list = self._concatenate_context_reprs(bidaf_context_reprs,
                                                                                                 bidaf_context_vecs,
                                                                                                 bidaf_context_masks)

        # List of (D) shaped vectors
        ques_final_state_list = [ques_encoded_final_state[i] for i in range(0, batch_size)]

        batch_denotations = []
        for i in range(0, batch_size):
            # (D)
            qrepr = self._dropout(ques_final_state_list[i])
            # (C, D)
            context_repr = self._dropout(context_vec_list[i])
            # Shape (C)
            question_context_similarity = self._bool_bilinear(qrepr, context_repr)
            best_score = torch.max(question_context_similarity)
            boolean_prob = torch.sigmoid(best_score)
            batch_denotations.append(boolean_prob)

        outputs: Dict[str, torch.Tensor] = {}

        if ans_grounding_dict is not None:
            denotation_loss = self._compute_loss(gold_ans_type=gold_ans_type,
                                                 ans_grounding_dict=ans_grounding_dict,
                                                 ans_grounding_mask=ans_grounding_mask,
                                                 predicted_expected_denotations=batch_denotations)
            outputs["denotation_loss"] = denotation_loss

            outputs["loss"] = denotation_loss

        if metadata is not None:
            outputs["metadata"] = metadata
        outputs["denotations"] = batch_denotations

        return outputs

    def _average_ques_repr(self, ques_reprs, ques_masks):
        """ Average question reprs from multiple runs of bidaf

        Parameters:
        -----------
        ques_reprs: `List[torch.FloatTensor]`
            A list of (B, T, D) sized tensors from multiple runs of bidaf for different contexts
        """
        avg_ques_repr = sum(ques_reprs)/float(len(ques_reprs))
        # Since mask for each run should be the same
        avg_ques_mask = ques_masks[0]

        return (avg_ques_repr, avg_ques_mask)

    def _concatenate_context_reprs(self, context_reprs, context_vecs, context_masks):
        """ Concatenate context_repr from same instance into a single tensor, and return a list of these for the batch
        Input is list of different batched context_repr from different runs of the bidaf model. Here we extract
        different contexts for the same instance and concatenate them.

        Parameters:
        -----------
        context_reprs: `List[torch.FloatTensor]`
            A C-sized list of (B, T, D) sized tensors
        context_vecs: `List[torch.FloatTensor]`
            A C-sized list of (B, D) sized tensors
        context_masks: `List[torch.FloatTensor]`
            A C-sized list of (B, T) sized tensors
        """

        num_contexts = len(context_reprs)
        batch_size = context_reprs[0].size()[0]

        # We will unsqueeze dim-1 of all input tensors, concatenate it along that axis to make single tensor of
        # (B, C, T, D) and then slice instances out of it

        context_reprs_expanded = [t.unsqueeze(1) for t in context_reprs]
        context_vecs_expanded = [t.unsqueeze(1) for t in context_vecs]
        context_masks_expanded = [t.unsqueeze(1) for t in context_masks]
        context_reprs_concatenated = torch.cat(context_reprs_expanded, dim=1)
        context_vecs_concatenated = torch.cat(context_vecs_expanded, dim=1)
        context_masks_concatenated = torch.cat(context_masks_expanded, dim=1)

        context_repr_list = [context_reprs_concatenated[i] for i in range(batch_size)]
        context_vec_list = [context_vecs_concatenated[i] for i in range(batch_size)]
        context_mask_list = [context_masks_concatenated[i] for i in range(batch_size)]

        return (context_repr_list, context_vec_list, context_mask_list)



    def _goldAnsDict_and_Masks(self,
                               gold_ans_type: List[str],
                               **kwargs):
        """ If gold answer types are given, then make a dict of answers and equivalent masks based on type.
        Also give a list of possible first actions when decode to make programs of valid types only.

        :param gold_ans_types:
        :return:
        """

        if gold_ans_type is None:
            return None, None

        # Field_name containing the grounding for type T is "ans_grounding_T"
        ans_grounding_prefix = "ans_grounding_"

        ans_grounding_dict = {}

        for k, v in kwargs.items():
            if k.startswith(ans_grounding_prefix):
                anstype = k[len(ans_grounding_prefix):]
                ans_grounding_dict[anstype] = v

        ans_grounding_mask = {}
        # Create masks for different types
        for ans_type, ans_grounding in ans_grounding_dict.items():
            if ans_type in types.ANS_TYPES:
                mask = (ans_grounding >= 0.0).float()
                ans_grounding_mask[ans_type] = mask

        return ans_grounding_dict, ans_grounding_mask


    def _compute_loss(self, gold_ans_type: List[str],
                            ans_grounding_dict: Dict,
                            ans_grounding_mask: Dict,
                            predicted_expected_denotations: List[torch.Tensor]):
        """ Compute the loss given the gold type, grounding, and predicted expected denotation of that type. """

        loss = 0.0

        # All answer denotations will comes as tensors
        # For each instance, compute expected denotation based on the prob of the action-seq.
        for ins_idx in range(0, len(predicted_expected_denotations)):
            gold_type = gold_ans_type[ins_idx]
            gold_denotation = ans_grounding_dict[gold_type][ins_idx]
            expected_denotation = predicted_expected_denotations[ins_idx]
            mask = ans_grounding_mask[gold_type][ins_idx]
            expected_denotation = expected_denotation * mask

            if gold_type == hpcons.BOOL_TYPE:
                # logger.info(f"Instance deno:\n{ins_denotations}")
                # logger.info(f"Mask:\n{mask}")
                # logger.info(f"Instance action probs:\n{instance_action_probs_ex}")
                # logger.info(f"Gold annotation:\n{gold_denotation}")
                # logger.info(f"Expected Deno:\n{expected_denotation}")
                instance_loss = F.binary_cross_entropy(input=expected_denotation, target=gold_denotation)

                loss += instance_loss

                bool_pred = (expected_denotation > 0.5).float()
                correct = 1.0 if (bool_pred == gold_denotation) else 0.0
                self.average_metric(float(correct))

        return loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self.average_metric.get_metric(reset)
        }
