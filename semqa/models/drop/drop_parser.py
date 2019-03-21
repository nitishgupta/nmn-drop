import logging
from typing import List, Dict, Any, Tuple, Optional
import math

from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, Attention, TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.nn import Activation
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.state_machines.states import GrammarBasedState
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models.reading_comprehension.util import get_best_span

from semqa.state_machines.constrained_beam_search import ConstrainedBeamSearch
from semqa.state_machines.transition_functions.linking_transition_func_emb import LinkingTransitionFunctionEmbeddings
from allennlp.training.metrics import Average, DropEmAndF1
from semqa.models.utils.bidaf_utils import PretrainedBidafModelUtils
from semqa.models.utils import semparse_utils

from semqa.models.drop.drop_parser_base import DROPParserBase
from semqa.domain_languages.drop import DropLanguage, Date, ExecutorParameters, QuestionSpanAnswer, PassageSpanAnswer

import datasets.drop.constants as dropconstants

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# In this three tuple, add "QENT:qent QSTR:qstr" in the twp gaps, and join with space to get the logical form
GOLD_BOOL_LF = ("(bool_and (bool_qent_qstr ", ") (bool_qent_qstr", "))")

def getGoldLF(qent1_action, qent2_action, qstr_action):
    qent1, qent2, qstr = qent1_action.split(' -> ')[1], qent2_action.split(' -> ')[1], qstr_action.split(' -> ')[1]
    # These qent1, qent2, and qstr are actions
    return f"{GOLD_BOOL_LF[0]} {qent1} {qstr}{GOLD_BOOL_LF[1]} {qent2} {qstr}{GOLD_BOOL_LF[2]}"


@Model.register("drop_parser")
class DROPSemanticParser(DROPParserBase):
    def __init__(self,
                 vocab: Vocabulary,
                 action_embedding_dim: int,
                 transitionfunc_attention: Attention,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 decoder_beam_search: ConstrainedBeamSearch,
                 max_decoding_steps: int,
                 goldactions: bool = None,
                 aux_goldprog_loss: bool = None,
                 qatt_coverage_loss: bool = None,
                 question_token_repr_key: str = None,
                 context_token_repr_key: str = None,
                 bidafutils: PretrainedBidafModelUtils = None,
                 dropout: float = 0.0,
                 text_field_embedder: TextFieldEmbedder = None,
                 debug: bool = False,
                 initializers: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        if bidafutils is not None:
            _text_field_embedder = bidafutils._bidaf_model._text_field_embedder
            phrase_layer = bidafutils._bidaf_model._phrase_layer
            matrix_attention_layer = bidafutils._bidaf_model._matrix_attention
            modeling_layer = bidafutils._bidaf_model._modeling_layer
            # TODO(nitish): explicity making text_field_embedder = None since it is initialized with empty otherwise
            text_field_embedder = None
        elif text_field_embedder is not None:
            _text_field_embedder = text_field_embedder
        else:
            _text_field_embedder = None
            raise NotImplementedError

        super(DROPSemanticParser, self).__init__(vocab=vocab,
                                                 action_embedding_dim=action_embedding_dim,
                                                 dropout=dropout,
                                                 debug=debug,
                                                 regularizer=regularizer)


        question_encoding_dim = phrase_layer.get_output_dim()
        self._decoder_step = LinkingTransitionFunctionEmbeddings(encoder_output_dim=question_encoding_dim,
                                                                 action_embedding_dim=action_embedding_dim,
                                                                 input_attention=transitionfunc_attention,
                                                                 num_start_types=1,
                                                                 activation=Activation.by_name('tanh')(),
                                                                 predict_start_type_separately=False,
                                                                 add_action_bias=False,
                                                                 dropout=dropout)

        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1
        # This metrircs measure accuracy of
        # (1) Top-predicted program, (2) ExpectedDenotation from the beam (3) Best accuracy from topK(5) programs
        self.top1_acc_metric = Average()
        self.expden_acc_metric = Average()
        self.topk_acc_metric = Average()
        self.aux_goldparse_loss = Average()
        self.qent_loss = Average()
        self.qattn_cov_loss_metric = Average()

        self._text_field_embedder = _text_field_embedder

        text_embed_dim = self._text_field_embedder.get_output_dim()
        encoding_in_dim = phrase_layer.get_input_dim()
        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)

        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)
        self._phrase_layer = phrase_layer
        self._matrix_attention = matrix_attention_layer
        self._modeling_layer = modeling_layer

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._executor_parameters = ExecutorParameters(num_highway_layers=num_highway_layers,
                                                       phrase_layer=phrase_layer,
                                                       matrix_attention_layer=matrix_attention_layer,
                                                       modeling_layer=modeling_layer,
                                                       hidden_dim=200)

        self._drop_metrics = DropEmAndF1()
        initializers(self)

    def device_id(self):
        allenutil.get_device_of()

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                passageidx2numberidx: torch.LongTensor,
                passage_number_values: List[int],
                passageidx2dateidx: torch.LongTensor,
                passage_date_values: List[List[Date]],
                # passage_number_indices: torch.LongTensor,
                # passage_number_entidxs: torch.LongTensor,
                # passage_date_spans: torch.LongTensor,
                # passage_date_entidxs: torch.LongTensor,
                actions: List[List[ProductionRule]],
                answer_types: List[str] = None,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                epoch_num: List[int] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size = len(actions)

        if epoch_num is not None:
            # epoch_num in allennlp starts from 0
            epoch = epoch_num[0] + 1
        else:
            epoch = None

        question_mask = allenutil.get_text_field_mask(question).float()
        passage_mask = allenutil.get_text_field_mask(passage).float()
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))
        embedded_question = self._highway_layer(self._embedding_proj_layer(embedded_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        projected_embedded_question = self._encoding_proj_layer(embedded_question)
        projected_embedded_passage = self._encoding_proj_layer(embedded_passage)

        # Shape: (batch_size, question_length, encoding_dim)
        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))

        """ Parser setup """
        # Shape: (B, encoding_dim)
        question_encoded_final_state = allenutil.get_final_encoder_states(encoded_question,
                                                                          question_mask,
                                                                          self._phrase_layer.is_bidirectional())
        question_encoded_aslist = [encoded_question[i] for i in range(batch_size)]
        question_mask_aslist = [question_mask[i] for i in range(batch_size)]
        passage_encoded_aslist = [encoded_passage[i] for i in range(batch_size)]
        passage_mask_aslist = [passage_mask[i] for i in range(batch_size)]

        # Based on the gold answer - figure out possible start types for the instance_language
        if answer_as_passage_spans is not None:
            batch_start_types = self.find_valid_start_states(answer_as_question_spans=answer_as_question_spans,
                                                             answer_as_passage_spans=answer_as_passage_spans,
                                                             batch_size=batch_size)
        else:
            batch_start_types = [None for _ in range(batch_size)]

        languages = [DropLanguage(encoded_question=question_encoded_aslist[i],
                                  encoded_passage=passage_encoded_aslist[i],
                                  question_mask=question_mask_aslist[i],
                                  passage_mask=passage_mask_aslist[i],
                                  passage_tokenidx2dateidx=passageidx2dateidx[i],
                                  passage_date_values=passage_date_values[i],
                                  parameters=self._executor_parameters,
                                  start_types=batch_start_types[i]) for i in range(batch_size)]

        # List[torch.Tensor(0.0)] -- Initial log-score list for the decoding
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for _ in range(batch_size)]

        initial_grammar_statelets = [self._create_grammar_statelet(languages[i], actions[i]) for i in range(batch_size)]

        initial_rnn_states = self._get_initial_rnn_state(question_encoded=encoded_question,
                                                         question_mask=question_mask,
                                                         question_encoded_finalstate=question_encoded_final_state,
                                                         question_encoded_aslist=question_encoded_aslist,
                                                         question_mask_aslist=question_mask_aslist)

        initial_side_args = [[] for _ in range(batch_size)]

        # Initial grammar state for the complete batch
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_states,
                                          grammar_state=initial_grammar_statelets,
                                          possible_actions=actions,
                                          debug_info=initial_side_args)

        # Mapping[int, Sequence[StateType]]
        best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                             initial_state,
                                                             self._decoder_step,
                                                             # firststep_allowed_actions=firststep_action_ids,
                                                             keep_final_unfinished_states=False)

        # batch_actionidxs: List[List[List[int]]]: All action sequence indices for each instance in the batch
        # batch_actionseqs: List[List[List[str]]]: All decoded action sequences for each instance in the batch
        # batch_actionseq_scores: List[List[torch.Tensor]]: Score for each program of each instance
        # batch_actionseq_probs: List[torch.FloatTensor]: Tensor containing normalized_prog_probs for each instance
        # batch_actionseq_sideargs: List[List[List[Dict]]]: List of side_args for each program of each instance
        # The actions here should be in the exact same order as passed when creating the initial_grammar_state ...
        # since the action_ids are assigned based on the order passed there.
        (batch_actionidxs,
         batch_actionseqs,
         batch_actionseq_scores,
         batch_actionseq_probs,
         batch_actionseq_sideargs) = semparse_utils._convert_finalstates_to_actions(best_final_states=best_final_states,
                                                                                    possible_actions=actions,
                                                                                    batch_size=batch_size)

        # List[List[Any]], List[List[str]]: Denotations and their types for all instances
        batch_denotations, batch_denotation_types = self._get_denotations(batch_actionseqs,
                                                                          languages,
                                                                          batch_actionseq_sideargs)

        # for i in range(len(batch_actionseqs)):
        #     for acseq in batch_actionseqs[i]:
        #         print(languages[i].action_sequence_to_logical_form(acseq))

        output_dict = {}
        ''' Computing losses if gold answers are given '''
        if answer_types is not None:
            loss = 0
            for i in range(batch_size):
                if answer_types[i] == dropconstants.SPAN_TYPE:
                    instance_prog_denotations, instance_prog_types = batch_denotations[i], batch_denotation_types[i]
                    # Tensor with shape: (num_of_programs, )
                    instance_prog_probs = batch_actionseq_probs[i]
                    instance_progs_logprob_list = batch_actionseq_scores[i]
                    instance_log_likelihood_list = []
                    for progidx in range(len(instance_prog_denotations)):
                        denotation = instance_prog_denotations[progidx]
                        progtype = instance_prog_types[progidx]
                        if progtype == "QuestionSpanAnswer":
                            denotation: QuestionSpanAnswer = denotation
                            log_likelihood = self._get_span_answer_log_prob(answer_as_spans=answer_as_question_spans[i],
                                                                            span_log_probs=denotation._value)
                            if torch.isnan(log_likelihood) == 1:
                                print("\nQuestionSpan")
                                print(denotation.start_logits)
                                print(denotation.end_logits)
                                print(denotation._value)
                        elif progtype == "PassageSpanAnswer":
                            denotation: PassageSpanAnswer = denotation
                            log_likelihood = self._get_span_answer_log_prob(answer_as_spans=answer_as_passage_spans[i],
                                                                            span_log_probs=denotation._value)
                            if torch.isnan(log_likelihood) == 1:
                                print("\nPassageSpan")
                                print(denotation.start_logits)
                                print(denotation.end_logits)
                                print(denotation._value)
                        else:
                            raise NotImplementedError

                        instance_log_likelihood_list.append(log_likelihood)

                    # Each is the shape of (number_of_progs,)
                    instance_denotation_log_likelihoods = torch.stack(instance_log_likelihood_list, dim=-1)
                    instance_progs_log_probs = torch.stack(instance_progs_logprob_list, dim=-1)

                    print()
                    print(f"IsstanceDenoLogProb:{instance_log_likelihood_list}")
                    print(f"InstanceProgLogProb:{instance_progs_log_probs}")

                    allprogs_log_marginal_likelihoods = instance_denotation_log_likelihoods + instance_progs_log_probs
                    instance_marginal_log_likelihood = allenutil.logsumexp(allprogs_log_marginal_likelihoods)
                    print(f"InstanceMarginalLogLike:{instance_marginal_log_likelihood}")

                    loss += -1.0*instance_marginal_log_likelihood
                else:
                    raise NotImplementedError
            print(f"Loss: {loss}")
            batch_loss = loss / batch_size
            print(f"BatchLoss: {batch_loss}")

            output_dict["loss"] = batch_loss

        if metadata is not None:
            batch_question_strs = [metadata[i]["original_question"] for i in range(batch_size)]
            batch_passage_strs = [metadata[i]["original_passage"] for i in range(batch_size)]
            batch_passage_token_offsets = [metadata[i]["passage_token_offsets"] for i in range(batch_size)]
            batch_question_token_offsets = [metadata[i]["question_token_offsets"] for i in range(batch_size)]
            answer_annotations = [metadata[i]["answer_annotation"] for i in range(batch_size)]
            question_num_tokens = [metadata[i]["question_num_tokens"] for i in range(batch_size)]
            passage_num_tokens = [metadata[i]["passage_num_tokens"] for i in range(batch_size)]
            (batch_best_spans, batch_predicted_answers) = self._get_best_spans(batch_denotations,
                                                                               batch_denotation_types,
                                                                               batch_question_token_offsets,
                                                                               batch_question_strs,
                                                                               batch_passage_token_offsets,
                                                                               batch_passage_strs,
                                                                               question_num_tokens,
                                                                               passage_num_tokens,
                                                                               question_mask_aslist,
                                                                               passage_mask_aslist)
            predicted_answers = [batch_predicted_answers[i][0] for i in range(batch_size)]


            if answer_annotations:
                for i in range(batch_size):
                    self._drop_metrics(predicted_answers[i], [answer_annotations[i]])

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}



    @staticmethod
    def _get_span_answer_log_prob(answer_as_spans: torch.LongTensor,
                                  span_log_probs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """ Compute the log_marginal_likelihood for the answer_spans given log_probs for start/end
            Compute log_likelihood (product of start/end probs) of each ans_span
            Sum the prob (logsumexp) for each span and return the log_likelihood

        Parameters:
        -----------
        answer: ``torch.LongTensor`` Shape: (number_of_spans, 2)
            These are the gold spans
        span_log_probs: ``torch.FloatTensor``
            2-Tuple with tensors of Shape: (length_of_sequence) for span_start/span_end log_probs

        Returns:
        log_marginal_likelihood_for_passage_span
        """
        # Unsqueezing dim=0 to make a batch_size of 1
        answer_as_spans = answer_as_spans.unsqueeze(0)

        span_start_log_probs, span_end_log_probs = span_log_probs
        span_start_log_probs = span_start_log_probs.unsqueeze(0)
        span_end_log_probs = span_end_log_probs.unsqueeze(0)

        # (batch_size, number_of_ans_spans)
        gold_passage_span_starts = answer_as_spans [:, :, 0]
        gold_passage_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_passage_span_mask = (gold_passage_span_starts != -1).long()
        clamped_gold_passage_span_starts = \
            allenutil.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
        clamped_gold_passage_span_ends = \
            allenutil.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = \
            torch.gather(span_start_log_probs, 1, clamped_gold_passage_span_starts)
        log_likelihood_for_span_ends = \
            torch.gather(span_end_log_probs, 1, clamped_gold_passage_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = \
            log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = \
            allenutil.replace_masked_values(log_likelihood_for_spans, gold_passage_span_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood_for_span = allenutil.logsumexp(log_likelihood_for_spans)

        return log_marginal_likelihood_for_span



    @staticmethod
    def find_valid_start_states(answer_as_question_spans: torch.LongTensor,
                                answer_as_passage_spans: torch.LongTensor,
                                batch_size: int):
        """ Firgure out valid start types based on gold answers
            If answer as question (passage) span exist, QuestionSpanAnswer (PassageSpanAnswer) are valid start types

        answer_as_question_spans: (B, N1, 2)
        answer_as_passage_spans: (B, N2, 2)

        Returns:
        --------
        start_types: `List[Set[Type]]`
            For each instance, a set of possible start_types
        """

        # List containing 0/1 indicating whether a question / passage answer span is given
        passage_span_ans_bool = [((answer_as_passage_spans[i] != -1).sum() > 0) for i in range(batch_size)]
        question_span_ans_bool = [((answer_as_question_spans[i] != -1).sum() > 0) for i in range(batch_size)]

        start_types = [set() for _ in range(batch_size)]
        for i in range(batch_size):
            if passage_span_ans_bool[i] > 0:
                start_types[i].add(PassageSpanAnswer)
            if question_span_ans_bool[i] > 0:
                start_types[i].add(QuestionSpanAnswer)

        return start_types


    @staticmethod
    def _get_best_spans(batch_denotations, batch_denotation_types,
                        question_char_offsets, question_strs, passage_char_offsets, passage_strs, *args):
        """ For all SpanType denotations, get the best span

        Parameters:
        ----------
        batch_denotations: List[List[Any]]
        batch_denotation_types: List[List[str]]
        """

        (question_num_tokens, passage_num_tokens, question_mask_aslist, passage_mask_aslist) = args

        batch_best_spans = []
        batch_predicted_answers = []

        for instance_idx in range(len(batch_denotations)):
            instance_prog_denotations = batch_denotations[instance_idx]
            instance_prog_types = batch_denotation_types[instance_idx]

            instance_best_spans = []
            instance_predicted_ans = []

            for denotation, progtype in zip(instance_prog_denotations, instance_prog_types):
                # if progtype == "QuestionSpanAnswwer":
                # Distinction between QuestionSpanAnswer and PassageSpanAnswer is not needed currently,
                # since both classes store the start/end logits as a tuple
                # Shape: (2, )
                best_span = get_best_span(span_end_logits=denotation._value[0].unsqueeze(0),
                                          span_start_logits=denotation._value[1].unsqueeze(0)).squeeze(0)
                instance_best_spans.append(best_span)

                predicted_span = tuple(best_span.detach().cpu().numpy())
                if progtype == "QuestionSpanAnswer":
                    try:
                        start_offset = question_char_offsets[instance_idx][predicted_span[0]][0]
                        end_offset = question_char_offsets[instance_idx][predicted_span[1]][1]
                        predicted_answer = question_strs[instance_idx][start_offset:end_offset]
                    except:
                        print()
                        print(f"PredictedSpan: {predicted_span}")
                        print(f"Question numtoksn: {question_num_tokens[instance_idx]}")
                        print(f"QuesMaskLen: {question_mask_aslist[instance_idx].size()}")
                        print(f"StartLogProbs:{denotation._value[0]}")
                        print(f"EndLogProbs:{denotation._value[1]}")
                        print(f"LenofOffsets: {len(question_char_offsets[instance_idx])}")
                        print(f"QuesStrLen: {len(question_strs[instance_idx])}")

                elif progtype == "PassageSpanAnswer":
                    try:
                        start_offset = passage_char_offsets[instance_idx][predicted_span[0]][0]
                        end_offset = passage_char_offsets[instance_idx][predicted_span[1]][1]
                        predicted_answer = passage_strs[instance_idx][start_offset:end_offset]
                    except:
                        print()
                        print(f"PredictedSpan: {predicted_span}")
                        print(f"Passagenumtoksn: {passage_num_tokens[instance_idx]}")
                        print(f"PassageMaskLen: {passage_mask_aslist[instance_idx].size()}")
                        print(f"LenofOffsets: {len(passage_char_offsets[instance_idx])}")
                        print(f"PassageStrLen: {len(passage_strs[instance_idx])}")
                else:
                    raise NotImplementedError

                instance_predicted_ans.append(predicted_answer)

            batch_best_spans.append(instance_best_spans)
            batch_predicted_answers.append(instance_predicted_ans)

        return batch_best_spans, batch_predicted_answers







