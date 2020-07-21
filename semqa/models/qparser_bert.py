import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union
import numpy as np
from overrides import overrides
import torch
import gc

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.data import TextFieldTensors
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules import Attention, Seq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn import Activation
from allennlp.modules.matrix_attention import DotProductMatrixAttention, LinearMatrixAttention
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average

from allennlp_semparse.state_machines.states import GrammarBasedState
from allennlp_semparse.state_machines.transition_functions import BasicTransitionFunction
from allennlp_semparse.state_machines.trainers.maximum_marginal_likelihood import MaximumMarginalLikelihood
from allennlp_semparse.state_machines import BeamSearch
from allennlp_semparse.state_machines import ConstrainedBeamSearch
from allennlp_semparse.fields.production_rule_field import ProductionRule
from semqa.modules.symbolic.utils import compute_token_symbol_alignments

from semqa.utils.rc_utils import DropEmAndF1
from semqa.state_machines.prefixed_beam_search import PrefixedConstrainedBeamSearch
from semqa.models.utils import semparse_utils
from semqa.utils import qdmr_utils
from semqa.utils.qdmr_utils import Node
from semqa.models.drop_parser_base import DROPParserBase
from semqa.domain_languages.drop_language import (
    DropLanguage,
    Date,
    CountNumber,
    get_empty_language_object
)
from semqa.domain_languages.drop_execution_parameters import ExecutorParameters
from semqa.modules.spans import SingleSpanAnswer, MultiSpanAnswer
from semqa.modules.qp_encodings import BertJointQPEncoding, BertIndependentQPEncoding
from semqa.modules.qrepr_module.qrepr_to_module import QReprModuleExecution
import datasets.drop.constants as dropconstants
from semqa.profiler.profile import Profile, profile_func_decorator

from semqa.modules.shared_substructure_module import shared_substructure_utils


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_representation(pad_token_id: int,
                       sep_token_id: int,
                       text_field_embedder: BasicTextFieldEmbedder,
                       question: TextFieldTensors = None,
                       passage: TextFieldTensors = None):
    """
    Parameters:
    -----------
    question:
        Question wordpiece indices from "pretrained_transformer" token_indexer (w/ [CLS] [SEP])
    passage:
        Passage wordpiece indices from "pretrained_transformer" token_indexer (w/ [CLS] [SEP])

    Returns: All returns have [CLS] and [SEP] removed
    --------
    question_token_idxs / passage_token_idxs::
        Question / Passage wordpiece indices after removing [CLS] and [SEP]
    question_mask / passage_mask:
        Question / Passage mask
    encoded_question / encoded_passage: `(batch_size, seq_length, BERT_dim)`
        Contextual BERT representations
    pooled_encoding: `(batch_size, BERT_dim)`
        Pooled Question representation used to prime program-decoder. [CLS] embedding from Question bert-encoding
    """

    def get_token_ids_and_mask(text_field_tensors: TextFieldTensors):
        """Get token_idxs and mask for a BERT TextField. """
        # Removing [CLS] and last [SEP], there might still be [SEP] for shorter texts
        token_idxs = text_field_tensors["tokens"]["token_ids"][:, 1:-1]
        # mask for [SEP] and [PAD] tokens
        mask = (token_idxs != pad_token_id) * (token_idxs != sep_token_id)
        # Mask [SEP] and [PAD] within the question
        token_idxs = token_idxs * mask
        mask = mask.float()
        return token_idxs, mask

    question_bert_out = text_field_embedder(question)
    bert_pooled_out = question_bert_out[:, 0, :]  # CLS embedding
    # Remove [CLS] and last [SEP], and mask internal [SEP] and [PAD]
    question_token_idxs, question_mask = get_token_ids_and_mask(question)
    # Skip [CLS] and last [SEP]
    encoded_question = question_bert_out[:, 1:-1, :] * question_mask.unsqueeze(-1)

    output_dict = {
            "question_token_idxs": question_token_idxs,
            "question_mask": question_mask,
            "encoded_question": encoded_question,
            "pooled_encoding": bert_pooled_out}

    if passage is not None:
        passage_bert_out = text_field_embedder(passage)
        passage_token_idxs, passage_mask = get_token_ids_and_mask(passage)
        encoded_passage = passage_bert_out[:, 1:-1, :] * passage_mask.unsqueeze(-1)

        output_dict.update(
            {
                "passage_token_idxs": passage_token_idxs,
                "passage_mask": passage_mask,
                "encoded_passage": encoded_passage}
        )

    return output_dict

@Model.register("ques_qparser_bert")
class DROPQuesParserBert(DROPParserBase):
    def __init__(
        self,
        vocab: Vocabulary,
        max_ques_len: int,
        action_embedding_dim: int,
        transitionfunc_attention: Attention,
        beam_size: int,
        max_decoding_steps: int,
        transformer_model_name: str,
        qattloss: bool = True,
        dropout: float = 0.0,
        profile_freq: Optional[int] = None,
        initializers: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super(DROPQuesParserBert, self).__init__(
            vocab=vocab,
            action_embedding_dim=action_embedding_dim,
            dropout=dropout,
            debug=False,
            regularizer=regularizer,
        )

        self._text_field_embedder: BasicTextFieldEmbedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(transformer_model_name)}
        )

        self._allennlp_tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model_name)
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._pad_token_id = self._tokenizer.pad_token_id
        self._sep_token_id = self._tokenizer.sep_token_id

        self.max_ques_len = max_ques_len

        question_encoding_dim = self._text_field_embedder.get_output_dim()
        self.bert_dim = question_encoding_dim

        self._decoder_step = BasicTransitionFunction(
            encoder_output_dim=question_encoding_dim,
            action_embedding_dim=action_embedding_dim,
            input_attention=transitionfunc_attention,
            activation=Activation.by_name("tanh")(),
            add_action_bias=False,
            dropout=dropout,
        )
        self._mml = MaximumMarginalLikelihood()

        self._beam_size = beam_size
        self._decoder_beam_search = BeamSearch(beam_size=self._beam_size)
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1

        # self.passage_token_to_date = passage_token_to_date
        self.dotprod_matrix_attn = DotProductMatrixAttention()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self.qattloss = qattloss
        if not self.qattloss:
            logger.warning("************    Training without question-attention loss   ********************")

        self.parseacc_metric = Average()
        self.qattloss_metric = Average()
        self.mmlloss_metric = Average()

        initializers(self)

        self.profile_steps = 0
        self.profile_freq = None if profile_freq == 0 else profile_freq
        self.device_id = None
        self.gc_steps = 0

    @profile_func_decorator
    @overrides
    def forward(
        self,
        question: TextFieldTensors,
        actions: List[List[ProductionRule]],
        question_passage: TextFieldTensors = None,
        answer_program_start_types: List[Union[List[str], None]] = None,
        program_supervised: List[bool] = None,   # TODO(nitishg): All train progs should be supervised; not needed
        gold_action_seqs: List[Tuple[List[List[int]], List[List[int]]]] = None,
        gold_function2actionidx_maps: List[List[List[int]]] = None,
        gold_program_dicts: List[List[Union[Dict, None]]] = None,
        gold_program_lisps: List[List[str]] = None,
        epoch_num: List[int] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        self.gc_steps += 1
        if self.gc_steps % 500 == 0:
            gc.collect()

        if self.device_id is None:
            self.device_id = allenutil.get_device_of(self._decoder_step._input_projection_layer.weight.data)
        batch_size = len(actions)

        self.profile_steps += 1
        if self.profile_freq is not None:
            if self.profile_steps % self.profile_freq == 0:
                logger.info(Profile.to_string())

        qp_encoder_output = get_representation(pad_token_id=self._pad_token_id,
                                               sep_token_id=self._sep_token_id,
                                               text_field_embedder=self._text_field_embedder,
                                               question=question, passage=None)

        question_token_idxs = qp_encoder_output["question_token_idxs"]
        question_mask = qp_encoder_output["question_mask"]
        encoded_question = qp_encoder_output["encoded_question"]
        bert_pooled_out = qp_encoder_output["pooled_encoding"]

        # epoch_num in AllenNLP starts from 0
        epoch = epoch_num[0] + 1 if epoch_num is not None else None

        """ Parser setup """
        # Shape: (B, encoding_dim)
        question_encoded_final_state = bert_pooled_out
        question_encoded_aslist = [encoded_question[i] for i in range(batch_size)]
        question_mask_aslist = [question_mask[i] for i in range(batch_size)]

        with Profile("lang_init"):
            languages = [
                get_empty_language_object()
                # DropLanguage(
                #     encoded_passage=passage_encoded_aslist[i],
                #     modeled_passage=passage_modeled_aslist[i],
                #     passage_mask=passage_mask[i],  # passage_mask_aslist[i],
                #     passage_sentence_boundaries=p_sentboundary_wps[i],
                #     passage_tokenidx2dateidx=passageidx2dateidx[i],
                #     passage_date_values=passage_date_values[i],
                #     passage_tokenidx2numidx=passageidx2numberidx[i],
                #     passage_num_values=passage_number_values[i],
                #     composed_numbers=composed_numbers[i],
                #     passage_number_sortedtokenidxs=passage_number_sortedtokenidxs[i],
                #     add_num_combination_indices=add_num_combination_aslist[i],
                #     sub_num_combination_indices=sub_num_combination_aslist[i],
                #     year_differences=year_differences[i],
                #     year_differences_mat=year_differences_mat[i],
                #     count_num_values=count_values[i],
                #     passage_token2date_alignment=p2pdate_alignment_aslist[i],
                #     passage_token2startdate_alignment=p2pstartdate_alignment_aslist[i],
                #     passage_token2enddate_alignment=p2penddate_alignment_aslist[i],
                #     passage_token2num_alignment=p2pnum_alignment_aslist[i],
                #     parameters=self._executor_parameters,
                #     start_types=None,  # batch_start_types[i],
                #     device_id=self.device_id,
                #     debug=self._debug,
                #     metadata=metadata[i],
                # )
                for i in range(batch_size)
            ]

            # This works because all instance-languages have the same action space
            action2idx_map = {rule: i for i, rule in enumerate(languages[0].all_possible_productions())}

        action_indices = list(action2idx_map.values())
        batch_action_indices: List[List[int]] = [action_indices for _ in range(batch_size)]

        """
        While training, we know the correct start-types for all instances and the gold-programs for some.
        For instances,
            #   with gold-programs, we should run a ConstrainedBeamSearch with target_sequences,
            #   with start-types, figure out the valid start-action-ids and run ConstrainedBeamSearch with firststep_allo..
        During Validation, we should **always** be running an un-constrained BeamSearch on the full language
        """

        # print(qtypes)
        # print(program_supervised)
        # print(gold_action_seqs)

        if not all(program_supervised):
            logger.warning("All instances are not provided with program supervision")

        mml_loss = 0
        if self.training:
            # If any instance is provided with goldprog, we need to divide the batch into supervised / unsupervised
            # and run fully-constrained decoding on supervised, and start-type-constrained-decoding on the rest
            if any(program_supervised):
                supervised_instances = [i for (i, ss) in enumerate(program_supervised) if ss is True]
                unsupervised_instances = [i for (i, ss) in enumerate(program_supervised) if ss is False]

                # List of (gold_actionseq_idxs, gold_actionseq_masks) -- for supervised instances
                supervised_gold_actionseqs = self._select_indices_from_list(gold_action_seqs, supervised_instances)
                s_gold_actionseq_idxs, s_gold_actionseq_masks = zip(*supervised_gold_actionseqs)
                s_gold_actionseq_idxs = list(s_gold_actionseq_idxs)
                s_gold_actionseq_masks = list(s_gold_actionseq_masks)
                (supervised_initial_state, _, _) = self.initialState_forInstanceIndices(
                    supervised_instances,
                    languages,
                    actions,
                    encoded_question,
                    question_mask,
                    question_encoded_final_state,
                    question_encoded_aslist,
                    question_mask_aslist,
                )
                constrained_search = ConstrainedBeamSearch(
                    self._beam_size,
                    allowed_sequences=s_gold_actionseq_idxs,
                    allowed_sequence_mask=s_gold_actionseq_masks,
                )

                supervised_final_states = constrained_search.search(
                    initial_state=supervised_initial_state, transition_function=self._decoder_step
                )
                # Computing MML Loss
                for instance_states in supervised_final_states.values():
                    scores = [state.score[0].view(-1) for state in instance_states]
                    mml_loss += -allenutil.logsumexp(torch.cat(scores))
                mml_loss = mml_loss / len(supervised_final_states)

                if len(unsupervised_instances) > 0:
                    raise NotImplementedError
                else:
                    unsup_final_states = []

                # Merge final_states for supervised and unsupervised instances
                best_final_states = self.merge_final_states(
                    supervised_final_states, unsup_final_states, supervised_instances, unsupervised_instances
                )
            else:
                raise NotImplementedError
        else:
            # Prediction-mode; Run beam-search
            (initial_state, _, _) = self.getInitialDecoderState(
                languages,
                actions,
                encoded_question,
                question_mask,
                question_encoded_final_state,
                question_encoded_aslist,
                question_mask_aslist,
                batch_size,
            )
            # This is unconstrained beam-search
            best_final_states = self._decoder_beam_search.search(
                self._max_decoding_steps, initial_state, self._decoder_step, keep_final_unfinished_states=False
            )

        # batch_actionidxs: List[List[List[int]]]: All action sequence indices for each instance in the batch
        # batch_actionseqs: List[List[List[str]]]: All decoded action sequences for each instance in the batch
        # batch_actionseq_scores: List[List[torch.Tensor]]: Score for each program of each instance
        # batch_actionseq_probs: List[torch.FloatTensor]: Tensor containing normalized_prog_probs for each instance - no longer
        # batch_actionseq_sideargs: List[List[List[Dict]]]: List of side_args for each program of each instance
        # The actions here should be in the exact same order as passed when creating the initial_grammar_state ...
        # since the action_ids are assigned based on the order passed there.
        (
            batch_actionidxs,
            batch_actionseqs,
            batch_actionseq_logprobs,
            batch_actionseq_sideargs,
        ) = semparse_utils._convert_finalstates_to_actions(
                best_final_states=best_final_states, possible_actions=actions, batch_size=batch_size
        )
        batch_actionseq_probs = [[torch.exp(logprob) for logprob in instance_programs]
                                 for instance_programs in batch_actionseq_logprobs]

        self.add_aux_supervision_to_program_side_args(languages=languages,
                                                      batch_action_seqs=batch_actionseqs,
                                                      batch_actionseq_sideargs=batch_actionseq_sideargs,
                                                      program_supervised=program_supervised,
                                                      gold_program_dicts=gold_program_dicts,
                                                      gold_function2actionidx_maps=gold_function2actionidx_maps)

            # self.qrepr_module_exec.add_weighted_question_vector_to_sideargs(
            #     batch_action_seqs=batch_actionseqs,
            #     batch_actionseq_sideargs=batch_actionseq_sideargs,
            #     text_field_embedder=self._text_field_embedder,
            #     question_encoding=encoded_question,
            #     question_mask=question_mask,
            #     question=question,
            #     question_passage=question_passage,
            #     max_ques_len=self.max_ques_len,
            #     device_id=self.device_id,
            #     question_attn_mask_threshold=0.1)

        # PRINT PRED PROGRAMS
        # for idx, instance_progs in enumerate(batch_actionseqs):
        #     print(f"--------  InstanceIdx: {idx}  ----------")
        #     print(metadata[idx]["question"])
        #     probs = batch_actionseq_probs[idx]
        #     logprobs = batch_actionseq_logprobs[idx]
        #     start_types = answer_program_start_types[idx]
        #     numcomposition_types = composed_num_ans_composition_types[idx]
        #     print(f"AnsTypes: {start_types} \t CompositionFunctions:{numcomposition_types}")
        #     for prog, prob, logprob in zip(instance_progs, probs, logprobs):
        #         print(f"{prob}: {languages[idx].action_sequence_to_logical_form(prog)} -- {logprob}")
        # print()
        # import pdb
        # pdb.set_trace()

        output_dict = {}
        # Computing losses if gold answers are given
        total_aux_loss = allenutil.move_to_device(torch.tensor(0.0), self.device_id).float()

        if self.qattloss:
            # Compute Question Attention Supervision auxiliary loss
            qattn_loss = self._ques_attention_loss(batch_actionseq_sideargs, question_mask, metadata)
            # print(qattn_loss)
            if qattn_loss != 0.0:
                self.qattloss_metric(qattn_loss.item())
            total_aux_loss += qattn_loss

        # This is computed above during beam search
        if mml_loss != 0:
            self.mmlloss_metric(mml_loss.item())

        if torch.isnan(total_aux_loss):
            logger.info(f"TotalAuxLoss is nan.")
            total_aux_loss = 0.0

        output_dict["loss"] = mml_loss + total_aux_loss

        output_dict["metadata"] = metadata

        if not self.training:
            output_dict["top_action_seq"] = []
            output_dict["top_action_prob"] = []
            output_dict["top_action_siderags"] = []
            output_dict["top_action_lisp"] = []
            output_dict["gold_program_dict"] = []
            for i in range(batch_size):
                correct = False
                if len(batch_actionseqs[i]) > 0:
                    # try:
                    predicted_program_lisp = languages[i].action_sequence_to_logical_form(batch_actionseqs[i][0])
                    output_dict["top_action_seq"].append(batch_actionseqs[i][0])
                    output_dict["top_action_prob"].append(torch.exp(batch_actionseq_logprobs[i][0]))
                    output_dict["top_action_siderags"].append(batch_actionseq_sideargs[i][0])
                    output_dict["top_action_lisp"].append(predicted_program_lisp)
                    # except:
                    #     predicted_program_lisp = ""
                    #     output_dict["top_action_seq"].append([])
                    #     output_dict["top_action_prob"].append(0.0)
                    gold_lisp = gold_program_lisps[i][0]
                    correct = (gold_lisp == predicted_program_lisp)
                else:
                    output_dict["top_action_seq"].append([])
                    output_dict["top_action_prob"].append(0.0)
                    output_dict["top_action_siderags"].append([])
                    output_dict["top_action_lisp"].append("")
                self.parseacc_metric(int(correct))

                output_dict["gold_program_dict"].append(gold_program_dicts[i][0])

        # Get the predicted answers irrespective of loss computation.
        # For each program, given it's return type compute the predicted answer string
        # if metadata is not None:
        #     output_dict["predicted_answer"] = []
        #     output_dict["all_predicted_answers"] = []
        #
        #     for i in range(batch_size):
        #         original_question = metadata[i]["question"]
        #         original_passage = metadata[i]["passage"]
        #         passage_wp_offsets = metadata[i]["passage_wp_offsets"]
        #         passage_token_charidxs = metadata[i]["passage_token_charidxs"]
        #         passage_tokens = metadata[i]["passage_tokens"]
        #         p_tokenidx2wpidx = metadata[i]['passage_tokenidx2wpidx']
        #         instance_year_differences = year_differences[i]
        #         instance_passage_numbers = passage_number_values[i]
        #         instance_composed_numbers= composed_numbers[i]
        #         instance_count_values = count_values[i]
        #         instance_prog_denotations, instance_prog_types = (batch_denotations[i], batch_denotation_types[i])
        #
        #         all_instance_progs_predicted_answer_strs: List[str] = []  # List of answers from diff instance progs
        #         for progidx in range(len(instance_prog_denotations)):
        #             denotation = instance_prog_denotations[progidx]
        #             progtype = instance_prog_types[progidx]
        #             if progtype == "PassageSpanAnswer":
        #                 span_answer_decode_inputs = {}
        #                 if not self.bio_tagging:
        #                     # Single-span predictions are made based on word-pieces
        #                     span_answer_decode_inputs.update({
        #                         "span_start_logits": denotation.start_logits,
        #                         "span_end_logits": denotation.end_logits,
        #                         "passage_token_offsets": passage_wp_offsets,
        #                         "passage_text": original_passage,
        #                     })
        #                 else:
        #                     span_answer_decode_inputs.update({
        #                         'log_probs': denotation.bio_logprobs,
        #                         'passage_mask': passage_mask[i, :],
        #                         'p_text': original_passage,
        #                         'p_tokenidx2wpidx': p_tokenidx2wpidx,
        #                         'passage_token_charidxs': passage_token_charidxs,
        #                         'passage_tokens': passage_tokens,
        #                         'original_question': original_question,
        #                     })
        #
        #                 predicted_answer: Union[str, List[str]] = self.span_answer.decode_answer(
        #                     **span_answer_decode_inputs)
        #
        #             elif progtype == "QuestionSpanAnswer":
        #                 raise NotImplementedError
        #             elif progtype == "YearDifference":
        #                 # Distribution over year_differences vector
        #                 year_differences_dist = denotation._value.detach().cpu().numpy()
        #                 predicted_yeardiff_idx = np.argmax(year_differences_dist)
        #                 # If not predicting year_diff = 0
        #                 # if predicted_yeardiff_idx == 0 and len(instance_year_differences) > 1:
        #                 #     predicted_yeardiff_idx = np.argmax(year_differences_dist[1:])
        #                 #     predicted_yeardiff_idx += 1
        #                 predicted_year_difference = instance_year_differences[predicted_yeardiff_idx]  # int
        #                 predicted_answer = str(predicted_year_difference)
        #             elif progtype == "PassageNumber":
        #                 predicted_passagenum_idx = torch.argmax(denotation._value).detach().cpu().numpy()
        #                 predicted_passage_number = instance_passage_numbers[predicted_passagenum_idx]  # int/float
        #                 predicted_passage_number = (
        #                     int(predicted_passage_number)
        #                     if int(predicted_passage_number) == predicted_passage_number
        #                     else predicted_passage_number
        #                 )
        #                 predicted_answer = str(predicted_passage_number)
        #             elif progtype == "ComposedNumber":
        #                 predicted_composednum_idx = torch.argmax(denotation._value).detach().cpu().numpy()
        #                 predicted_composed_number = instance_composed_numbers[predicted_composednum_idx]  # int/float
        #                 predicted_composed_number = (
        #                     int(predicted_composed_number)
        #                     if int(predicted_composed_number) == predicted_composed_number
        #                     else predicted_composed_number
        #                 )
        #                 predicted_answer = str(predicted_composed_number)
        #             elif progtype == "CountNumber":
        #                 denotation: CountNumber = denotation
        #                 count_idx = torch.argmax(denotation._value).detach().cpu().numpy()
        #                 predicted_count_answer = instance_count_values[count_idx]
        #                 predicted_count_answer = (
        #                     int(predicted_count_answer)
        #                     if int(predicted_count_answer) == predicted_count_answer
        #                     else predicted_count_answer
        #                 )
        #                 predicted_answer = str(predicted_count_answer)
        #
        #             else:
        #                 raise NotImplementedError
        #
        #             all_instance_progs_predicted_answer_strs.append(predicted_answer)
        #         # If no program was found in beam-search
        #         if len(all_instance_progs_predicted_answer_strs) == 0:
        #             all_instance_progs_predicted_answer_strs.append("")
        #         # Since the programs are sorted by decreasing scores, we can directly take the first pred answer
        #         instance_predicted_answer = all_instance_progs_predicted_answer_strs[0]
        #         output_dict["predicted_answer"].append(instance_predicted_answer)
        #         output_dict["all_predicted_answers"].append(all_instance_progs_predicted_answer_strs)
        #
        #         answer_annotations = metadata[i].get("answer_annotations", [])
        #         if answer_annotations:
        #             self._drop_metrics(instance_predicted_answer, answer_annotations)
        #
        #     if not self.training: # and self._debug:
        #         output_dict["metadata"] = metadata
        #         output_dict["question_mask"] = question_mask
        #         output_dict["passage_mask"] = passage_mask
        #         output_dict["passage_token_idxs"] = passage_token_idxs
        #         # if answer_as_passage_spans is not None:
        #         #     output_dict["answer_as_passage_spans"] = answer_as_passage_spans
        #         output_dict["batch_action_seqs"] = batch_actionseqs
        #         batch_logical_programs = []
        #         for instance_idx, instance_actionseqs in enumerate(batch_actionseqs):
        #             instance_logical_progs = []
        #             for ins_actionseq in instance_actionseqs:
        #                 logical_form = languages[instance_idx].action_sequence_to_logical_form(ins_actionseq)
        #                 instance_logical_progs.append(logical_form)
        #             batch_logical_programs.append(instance_logical_progs)
        #
        #         output_dict["logical_forms"] = batch_logical_programs
        #         output_dict["actionseq_logprobs"] = batch_actionseq_logprobs
        #         output_dict["actionseq_probs"] = batch_actionseq_probs
        #         modules_debug_infos = [l.modules_debug_info for l in languages]
        #         output_dict["modules_debug_infos"] = modules_debug_infos

        return output_dict

    def compute_avg_norm(self, tensor):
        dim0_size = tensor.size()[0]
        dim1_size = tensor.size()[1]

        tensor_norm = tensor.norm(p=2, dim=2).sum() / (dim0_size * dim1_size)

        return tensor_norm

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        qatt_loss = self.qattloss_metric.get_metric(reset)
        mml_loss = self.mmlloss_metric.get_metric(reset)
        parseacc = self.parseacc_metric.get_metric(reset)
        metric_dict.update(
            {
                "acc": parseacc,
                "qatt": qatt_loss,
                "mml": mml_loss,
            }
        )

        return metric_dict

    @staticmethod
    def get_valid_start_actionids(answer_types: List[List[str]], action2actionidx: Dict[str, int],
                                  valid_numcomposition_types: List[Set[str]] = None) -> List[List[List[int]]]:
        """ For each instance, given answer_types return the set of valid start action ids that return an object of
            that type.
            For example, given start_type as 'PassageSpanAnswer', '@start@ -> PassageSpanAnswer' is a valid start action
            See the reader for all possible values of start_types

            This is used while *training* to run constrained-beam-search to only search through programs for which
            the gold answer is known. These *** shouldn't *** be used during validation
        Parameters:
        -----------
        answer_types: ``List[List[str]]``
            List of answer-types for each instance. These should be (a) possible start-types of the DomainLanguage (DL)
            and (b) "@start@ -> TYPE" action should exist for any TYPE, i.e. they should be types of the DL.
        action2actionidx:
            Mapping action-strings to action-idxs.
        valid_numcomposition_types:
            If answer-type is ComposedNumber, then the function (e.g. add or subtract) that would lead to the answer.
            This helps us constrain the program-search even further
        composed_num_ans_types: ``List[Set[str]]``
            For each instance, if the answer is of ComposedNumber type, then this tells valid compositions.
            Can be used to limit search space
        Returns:
        --------
        start_types: `List[Set[int]]`
            For each instance, a set of possible start_types
        """

        # This is the next action to go from ComposedNumber to a function that could generate it
        composed_num_function_action = "ComposedNumber -> [<PassageNumber,PassageNumber:ComposedNumber>, PassageNumber, PassageNumber]"
        # These two actions satisfies the function signature in composed_num_function_action
        addition_function_action = "<PassageNumber,PassageNumber:ComposedNumber> -> passagenumber_addition"
        subtraction_function_action = "<PassageNumber,PassageNumber:ComposedNumber> -> passagenumber_difference"

        # This is the key that the reader sends in valid_numcomposition_types
        num_composition_type_action_mapping = {
            "passage_num_addition": addition_function_action,
            "passage_num_subtraction": subtraction_function_action,
        }

        # We are aiming to make a List[List[List[int]]] -- for each instance, a list of prefix sequences.
        #  Each prefix sequence is a list of action indices. The prefixes don't need to be of the same length.
        #  If an instance does not have any prefixes, its prefix_sequences_list can remain empty
        action_prefixes: List[List[List[int]]] = []

        for i in range(len(answer_types)):
            instance_answer_types: List[str] = answer_types[i]
            instance_prefix_sequences: List[List[int]] = []
            for start_type in instance_answer_types:
                start_action = "@start@ -> {}".format(start_type)
                start_action_idx = action2actionidx[start_action]
                if start_type == "ComposedNumber":
                    # Containing "passage_num_addition" and/or "passage_num_subtraction"
                    instance_numcomposition_types: Set[str] = valid_numcomposition_types[i]
                    assert len(instance_numcomposition_types) > 0, "No composition type info given"
                    for composition_type in instance_numcomposition_types:
                        composition_function_action = num_composition_type_action_mapping[composition_type]
                        composition_function_action_idx = action2actionidx[composition_function_action]
                        composednumtype_to_funcsign_actionidx = action2actionidx[composed_num_function_action]
                        prefix_sequence = [start_action_idx, composednumtype_to_funcsign_actionidx,
                                           composition_function_action_idx]
                        instance_prefix_sequences.append(prefix_sequence)
                else:
                    prefix_sequence = [start_action_idx]
                    instance_prefix_sequences.append(prefix_sequence)
                # else:
                #     logging.error(f"StartType: {start_type} not present in {answer_type_to_start_action_mapping}")
            # valid_start_action_ids.append(start_action_ids)
            action_prefixes.append(instance_prefix_sequences)

        return action_prefixes

    def _ques_attention_loss(self,
                             batch_actionseq_sideargs: List[List[List[Dict]]], question_mask: torch.FloatTensor,
                             metadata):
        """Compute question attention loss against the `question_attention_supervision` key in side-args.

        Relevant actions should contain the `question_attention_supervision` key; compute loss against that.
        """
        loss = 0.0
        normalizer = 0
        for instance_idx in range(len(batch_actionseq_sideargs)):
            instance_actionseq_sideargs: List[List[Dict]] = batch_actionseq_sideargs[instance_idx]
            for program_sideargs in instance_actionseq_sideargs:
                for action_sideargs in program_sideargs:
                    if "question_attention_supervision" in action_sideargs:
                        gold_attn: List[int] = action_sideargs["question_attention_supervision"]
                        gold_attn_len = len(gold_attn)
                        gold_attn_tensor: torch.FloatTensor = allenutil.move_to_device(
                            torch.FloatTensor(gold_attn), cuda_device=self.device_id)
                        pred_attn = action_sideargs["question_attention"]
                        pred_attn = pred_attn[0:gold_attn_len]
                        mask = question_mask[instance_idx, 0:gold_attn_len]
                        gold_attn_tensor = gold_attn_tensor * mask
                        pred_attn = pred_attn * mask
                        if torch.sum(gold_attn_tensor) > 0:     # without this we get NaN loss when gold-attn is 0
                            # attn_sum = torch.sum(pred_attn * gold_attn_tensor)
                            # loss += torch.log(attn_sum + 1e-40)
                            log_question_attention = torch.log(pred_attn + 1e-40) * mask
                            loss += torch.sum(log_question_attention * gold_attn_tensor)
                            normalizer += 1

        if normalizer == 0:
            return loss
        else:
            return -1 * (loss / normalizer)


    def getInitialDecoderState(
        self,
        languages: List[DropLanguage],
        actions: List[List[ProductionRule]],
        encoded_question: torch.FloatTensor,
        question_mask: torch.FloatTensor,
        question_encoded_final_state: torch.FloatTensor,
        question_encoded_aslist: List[torch.Tensor],
        question_mask_aslist: List[torch.Tensor],
        batch_size: int,
    ):
        # List[torch.Tensor(0.0)] -- Initial log-score list for the decoding
        initial_score_list = [encoded_question.new_zeros(1, dtype=torch.float) for _ in range(batch_size)]

        initial_grammar_statelets = []
        batch_action2actionidx: List[Dict[str, int]] = []
        # This is kind of useless, only needed for debugging in BasicTransitionFunction
        batch_actionidx2actionstr: List[List[str]] = []

        for i in range(batch_size):
            (grammar_statelet, action2actionidx, actionidx2actionstr) = self._create_grammar_statelet(
                languages[i], actions[i]
            )

            initial_grammar_statelets.append(grammar_statelet)
            batch_actionidx2actionstr.append(actionidx2actionstr)
            batch_action2actionidx.append(action2actionidx)

        initial_rnn_states = self._get_initial_rnn_state(
            question_encoded=encoded_question,
            question_mask=question_mask,
            question_encoded_finalstate=question_encoded_final_state,
            question_encoded_aslist=question_encoded_aslist,
            question_mask_aslist=question_mask_aslist,
        )

        initial_side_args = [[] for _ in range(batch_size)]

        # Initial grammar state for the complete batch
        initial_state = GrammarBasedState(
            batch_indices=list(range(batch_size)),
            action_history=[[] for _ in range(batch_size)],
            score=initial_score_list,
            rnn_state=initial_rnn_states,
            grammar_state=initial_grammar_statelets,
            possible_actions=actions,
            extras=batch_actionidx2actionstr,
            debug_info=initial_side_args,
        )

        return (initial_state, batch_action2actionidx, batch_actionidx2actionstr)

    def _select_indices_from_list(self, list, indices):
        new_list = [list[i] for i in indices]
        return new_list

    def initialState_forInstanceIndices(
        self,
        instances_list: List[int],
        languages: List[DropLanguage],
        actions: List[List[ProductionRule]],
        encoded_question: torch.FloatTensor,
        question_mask: torch.FloatTensor,
        question_encoded_final_state: torch.FloatTensor,
        question_encoded_aslist: List[torch.Tensor],
        question_mask_aslist: List[torch.Tensor],
    ):

        s_languages = self._select_indices_from_list(languages, instances_list)
        s_actions = self._select_indices_from_list(actions, instances_list)
        s_encoded_question = encoded_question[instances_list]
        s_question_mask = question_mask[instances_list]
        s_question_encoded_final_state = question_encoded_final_state[instances_list]
        s_question_encoded_aslist = self._select_indices_from_list(question_encoded_aslist, instances_list)
        s_question_mask_aslist = self._select_indices_from_list(question_mask_aslist, instances_list)

        num_instances = len(instances_list)

        return self.getInitialDecoderState(
            s_languages,
            s_actions,
            s_encoded_question,
            s_question_mask,
            s_question_encoded_final_state,
            s_question_encoded_aslist,
            s_question_mask_aslist,
            num_instances,
        )

    def merge_final_states(
        self,
        supervised_final_states,
        unsupervised_final_states,
        supervised_instances: List[int],
        unsupervised_instances: List[int],
    ):

        """ Supervised and unsupervised final_states are dicts with keys in order from 0 - len(dict)
            The final keys are the instances' batch index which is stored in (un)supervised_instances list
            i.e. index = supervised_instances[0] is the index of the instance in the original batch
            whose final state is now in supervised_final_states[0].
            Therefore the final_state should contain; final_state[index] = supervised_final_states[0]
        """

        if len(supervised_instances) == 0:
            return unsupervised_final_states

        if len(unsupervised_instances) == 0:
            return supervised_final_states

        batch_size = len(supervised_instances) + len(unsupervised_instances)

        final_states = {}

        for i in range(batch_size):
            if i in supervised_instances:
                idx = supervised_instances.index(i)
                state_value = supervised_final_states[idx]
                final_states[i] = state_value
            else:
                idx = unsupervised_instances.index(i)
                # Unsupervised instances go through beam search and not always is a program found for them
                # If idx does not exist in unsupervised_final_states, don't add in final_states
                # Only add a instance_idx if it exists in final states -- not all beam-searches result in valid-paths
                if idx in unsupervised_final_states:
                    state_value = unsupervised_final_states[idx]
                    final_states[i] = state_value

        return final_states

    def masking_blockdiagonal(self, passage_length, window, device_id):
        """ Make a (passage_length, passage_length) tensor M of 1 and -1 in which for each row x,
            M[x, y] = -1 if y < x - window or y > x + window, else it is 1.
            Basically for the x-th row, the [x-win, x+win] columns should be 1, and rest -1
        """

        # The lower and upper limit of token-idx that won't be masked for a given token
        lower = allenutil.get_range_vector(passage_length, device=device_id) - window
        upper = allenutil.get_range_vector(passage_length, device=device_id) + window
        lower = torch.clamp(lower, min=0, max=passage_length - 1)
        upper = torch.clamp(upper, min=0, max=passage_length - 1)
        lower_un = lower.unsqueeze(1)
        upper_un = upper.unsqueeze(1)

        # Range vector for each row
        lower_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)
        upper_range_vector = allenutil.get_range_vector(passage_length, device=device_id).unsqueeze(0)

        # Masks for lower and upper limits of the mask
        lower_mask = lower_range_vector >= lower_un
        upper_mask = upper_range_vector <= upper_un

        # Final-mask that we require
        inwindow_mask = (lower_mask == upper_mask).float()
        outwindow_mask = (lower_mask != upper_mask).float()
        return inwindow_mask, outwindow_mask

    def window_loss_numdate(self, passage_passage_alignment, passage_tokenidx_mask, inwindow_mask, outwindow_mask):
        """
        The idea is to first softmax the similarity_scores to get a distribution over the date/num tokens from each
        passage token.

        For each passage_token,
            -- increase the sum of prob for date/num tokens around it in a window (don't know which date/num is correct)
            -- decrease the prod of prob for date/num tokens outside the window (know that all date/num are wrong)

        Parameters:
        -----------
        passage_passage_similarity_scores: (batch_size, passage_length, passage_length)
            For each passage_token, a similarity score to other passage_tokens for data/num
            This should ideally, already be masked

        passage_tokenidx_mask: (batch_size, passage_length)
            Mask for tokens that are num/date

        inwindow_mask: (passage_length, passage_length)
            For row x, inwindow_mask[x, x-window : x+window] = 1 and 0 otherwise. Mask for a window around the token
        outwindow_mask: (passage_length, passage_length)
            Opposite of inwindow_mask. For each row x, the columns are 1 outside of a window around x
        """
        inwindow_mask = inwindow_mask.unsqueeze(0) * passage_tokenidx_mask.unsqueeze(1)
        inwindow_probs = passage_passage_alignment * inwindow_mask
        # This signifies that each token can distribute it's prob to nearby-date/num in anyway
        # Shape: (batch_size, passage_length)
        sum_inwindow_probs = inwindow_probs.sum(2)
        mask_sum = (inwindow_mask.sum(2) > 0).float()
        # Image a row where mask = 0, there sum of probs will be zero and we need to compute masked_log
        masked_sum_inwindow_probs = allenutil.replace_masked_values(sum_inwindow_probs, mask_sum.bool(),
                                                                    replace_with=1e-40)
        log_sum_inwindow_probs = torch.log(masked_sum_inwindow_probs + 1e-40) * mask_sum
        inwindow_likelihood = torch.sum(log_sum_inwindow_probs)
        if torch.sum(inwindow_mask) > 0:
            inwindow_likelihood = inwindow_likelihood / torch.sum(inwindow_mask)
        else:
            inwindow_likelihood = 0.0

        outwindow_mask = outwindow_mask.unsqueeze(0) * passage_tokenidx_mask.unsqueeze(1)
        # For tokens outside the window, increase entropy of the distribution. i.e. -\sum p*log(p)
        # Since we'd like to distribute the weight equally to things outside the window
        # Shape: (batch_size, passage_length, passage_length)
        outwindow_probs = passage_passage_alignment * outwindow_mask
        masked_outwindow_probs = allenutil.replace_masked_values(outwindow_probs, outwindow_mask.bool(),
                                                                 replace_with=1e-40)
        outwindow_probs_log = torch.log(masked_outwindow_probs + 1e-40) * outwindow_mask
        # Shape: (batch_length, passage_length)
        outwindow_negentropies = torch.sum(outwindow_probs * outwindow_probs_log)

        if torch.sum(outwindow_mask) > 0:
            outwindow_negentropies = outwindow_negentropies / torch.sum(outwindow_mask)
        else:
            outwindow_negentropies = 0.0

        # Increase inwindow likelihod and decrease outwindow-negative-entropy
        loss = -1 * inwindow_likelihood + outwindow_negentropies
        return loss

    def add_aux_supervision_to_program_side_args(self,
                                                 languages: List[DropLanguage],
                                                 batch_action_seqs: List[List[List[str]]],
                                                 batch_actionseq_sideargs: List[List[List[Dict]]],
                                                 program_supervised: List[bool],
                                                 gold_program_dicts: List[List[Union[Dict, None]]],
                                                 gold_function2actionidx_maps: List[List[List[int]]]):
        """ If instance has a gold-program, update the side-args of the predicted program (same as gold) with
            supervision dictionaries from the gold-program.
        """
        if not self.training:
            return

        for instance_idx in range(len(program_supervised)):
            ins_prog_supervised = program_supervised[instance_idx]
            if not ins_prog_supervised:
                continue
            # Currently we assume there is only one gold-program per instance and this gets decoded as top-program
            # in constrained-decoding
            language = languages[instance_idx]
            gold_program_node = qdmr_utils.node_from_dict(gold_program_dicts[instance_idx][0])
            gold_program_lisp = qdmr_utils.nested_expression_to_lisp(gold_program_node.get_nested_expression())
            gold_action_seq: List[str] = language.logical_form_to_action_sequence(gold_program_lisp)
            # TODO(nitish): Faced issue in next line in evaluate once. list index out of range
            pred_action_seq: List[str] = batch_action_seqs[instance_idx][0]
            pred_sideargs: List[Dict] = batch_actionseq_sideargs[instance_idx][0]
            if gold_action_seq == pred_action_seq:
                # These are the action-idxs which produce terminal-node functions; gold_function2actionidx_maps is
                # packed so that as multiple gold-programs are passed. we assume singe gold-program here
                function2actionidx_map: List[int] = gold_function2actionidx_maps[instance_idx][0]
                inorder_supervision_dicts: List[Dict] = qdmr_utils.get_inorder_supervision_list(gold_program_node)
                assert len(function2actionidx_map) == len(inorder_supervision_dicts), "each func. should have a supdict"
                # Append the appropriate pred_sideargs idxs with the supervision-dict
                for supervision_idx, action_idx in enumerate(function2actionidx_map):
                    # supervision_idx: range(0, num_of_functions) - index into inorder_supervision_dicts
                    # action_idx: range(0, len(pred_sideargs)) - index into the actions
                    pred_sideargs[action_idx].update(inorder_supervision_dicts[supervision_idx])









