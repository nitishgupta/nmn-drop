from typing import List, Tuple, Dict

from enum import Enum

import torch
from allennlp.nn.util import move_to_device
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.modules.text_field_embedders import TextFieldEmbedder


class QREPR_METHOD(Enum):
    attn = 1
    decont = 2
    decont_qp = 3


class QReprModuleExecution():
    # making a static member so that the paired-data loss function can use this
    actions_w_qattn = [
        "PassageNumber -> select_implicit_num",
        "PassageAttention -> select_passage",
        "<PassageAttention:PassageAttention> -> filter_passage",
        "<PassageAttention:PassageAttention> -> project_passage",
        "<PassageAttention:PassageNumber> -> select_num",
        "<PassageAttention,PassageAttention:PassageAttention> -> compare_date_gt",
        "<PassageAttention,PassageAttention:PassageAttention> -> compare_date_lt",
        "<PassageAttention,PassageAttention:PassageAttention> -> compare_num_gt",
        "<PassageAttention,PassageAttention:PassageAttention> -> compare_num_lt",
        "<PassageAttention,PassageAttention:YearDifference> -> year_difference_two_events",
        "<PassageAttention:YearDifference> -> year_difference_single_event",
        "<PassageAttention:PassageAttention> -> select_min_num",
        "<PassageAttention:PassageAttention> -> select_max_num",
    ]

    def __init__(self, qrepr_style: str, qp_encoding_style: str):
        self._qrepr_style = qrepr_style
        self._qp_encoding_style: str = qp_encoding_style

        # Functions:
        # select_passage, filter_passage, project_passage,
        # compare_num_lt, compare_num_gt, compare_date_gt, compare_date_lt,
        # year_difference_two_events, year_difference_single_event,
        # select_num, select_min_num, select_max_num, select_implicit_num

        # self.relevant_actions = [
        #     "PassageNumber -> select_implicit_num",
        #     "PassageAttention -> select_passage",
        #     "<PassageAttention:PassageAttention> -> filter_passage",
        #     "<PassageAttention:PassageAttention> -> project_passage",
        #     "<PassageAttention:PassageNumber> -> select_num",
        #     "<PassageAttention,PassageAttention:PassageAttention> -> compare_date_gt",
        #     "<PassageAttention,PassageAttention:PassageAttention> -> compare_date_lt",
        #     "<PassageAttention,PassageAttention:PassageAttention> -> compare_num_gt",
        #     "<PassageAttention,PassageAttention:PassageAttention> -> compare_num_lt",
        #     "<PassageAttention,PassageAttention:YearDifference> -> year_difference_two_events",
        #     "<PassageAttention:YearDifference> -> year_difference_single_event",
        #     "<PassageAttention:PassageAttention> -> select_min_num",
        #     "<PassageAttention:PassageAttention> -> select_max_num",
        # ]

        assert self._qrepr_style in ["attn", "decont", "decont_qp"]

        # Enum allows programmatic access to member-names as strings
        self._qrepr_method: QREPR_METHOD = QREPR_METHOD[self._qrepr_style]


    def add_weighted_question_vector_to_sideargs(self,
                                                 batch_action_seqs: List[List[List[str]]],
                                                 batch_actionseq_sideargs: List[List[List[Dict]]],
                                                 text_field_embedder: TextFieldEmbedder,
                                                 question_encoding: torch.FloatTensor,
                                                 question_mask: torch.FloatTensor,
                                                 question: TextFieldTensors,
                                                 question_passage: TextFieldTensors,
                                                 max_ques_len: int,
                                                 device_id: int,
                                                 question_attn_mask_threshold: float = None):

        if self._qrepr_method == QREPR_METHOD.attn:
            self._add_attn_weighted_qrepr(batch_action_seqs=batch_action_seqs,
                                          batch_actionseq_sideargs=batch_actionseq_sideargs,
                                          question_encoding=question_encoding,
                                          question_mask=question_mask)

        elif self._qrepr_method == QREPR_METHOD.decont:
            self._add_decont_ques_repr(batch_action_seqs=batch_action_seqs,
                                       batch_actionseq_sideargs=batch_actionseq_sideargs,
                                       text_field_embedder=text_field_embedder,
                                       question=question,
                                       device_id=device_id,
                                       question_attn_mask_threshold=question_attn_mask_threshold)

        elif self._qrepr_method == QREPR_METHOD.decont_qp:
            self._add_decont_quespassage_repr(batch_action_seqs=batch_action_seqs,
                                              batch_actionseq_sideargs=batch_actionseq_sideargs,
                                              text_field_embedder=text_field_embedder,
                                              question_passage=question_passage,
                                              max_ques_len=max_ques_len,
                                              question_attn_mask_threshold=question_attn_mask_threshold)
        else:
            raise NotImplementedError

    def _add_attn_weighted_qrepr(self,
                                 batch_action_seqs: List[List[List[str]]],
                                 batch_actionseq_sideargs: List[List[List[Dict]]],
                                 question_encoding: torch.FloatTensor,
                                 question_mask: torch.FloatTensor):

        for instance_idx in range(len(batch_action_seqs)):
            for progidx, (action_seq, sideargs) in enumerate(zip(batch_action_seqs[instance_idx],
                                                                 batch_actionseq_sideargs[instance_idx])):
                for actionidx, (action_str, sidearg_dict) in enumerate(zip(action_seq, sideargs)):
                    if action_str in QReprModuleExecution.actions_w_qattn:
                        question_attention = sidearg_dict["question_attention"]
                        # if "question_attention_supervision" in sidearg_dict:
                        #     import pdb
                        #     pdb.set_trace()
                        question_attention = question_attention * question_mask[instance_idx, :]
                        weighted_question_vector = torch.sum(
                            question_encoding[instance_idx, :, :] * question_attention.unsqueeze(1), dim=0)
                        sidearg_dict["weighted_question_vector"] = weighted_question_vector

    def _add_decont_ques_repr(self,
                              batch_action_seqs: List[List[List[str]]],
                              batch_actionseq_sideargs: List[List[List[Dict]]],
                              text_field_embedder: TextFieldEmbedder,
                              question: TextFieldTensors,
                              device_id: int,
                              question_attn_mask_threshold: float = 0.1):
        """ Compute decontextualized question representation only using question as an input.

        Note:
            Shouldn't be used when computing a joint ques-passage encoding; it would render decontextualization useless
        """

        # List of (instance_idx, prog_idx, action_idx, question_attention) -- question-attention for relevant actions
        insidxprogaction_qattn = []

        # For all relevant actions, extract the (instance_idx, prog_idx, action_idx, question_attn)
        for instance_idx in range(len(batch_action_seqs)):
            for progidx, (action_seq, sideargs) in enumerate(zip(batch_action_seqs[instance_idx],
                                                                 batch_actionseq_sideargs[instance_idx])):
                for actionidx, (action_str, sidearg_dict) in enumerate(zip(action_seq, sideargs)):
                    if action_str in QReprModuleExecution.actions_w_qattn:
                        question_attention = sidearg_dict["question_attention"]
                        """ The code block below can be used to make model use gold-supervision instead of pred attn.
                        # if "question_attention_supervision" in sidearg_dict:
                        #     question_attn_len = question_attention.size()[0]
                        #     # This is of question-len without padding
                        #     question_attn_sup = move_to_device(
                        #         torch.FloatTensor(sidearg_dict["question_attention_supervision"]),
                        #         cuda_device=device_id)
                        #     actual_question_len = question_attn_sup.size()[0]
                        #     attn_sup = move_to_device(torch.zeros(question_attn_len), cuda_device=device_id)
                        #     attn_sup[0:actual_question_len] += question_attn_sup
                        #     insidxprogaction_qattn.append((instance_idx, progidx, actionidx, attn_sup))
                        # else:
                        #     insidxprogaction_qattn.append((instance_idx, progidx, actionidx, question_attention))
                        """
                        insidxprogaction_qattn.append((instance_idx, progidx, actionidx, question_attention))

        if len(insidxprogaction_qattn) == 0:
            # No relevant actions which need question-repr.
            return

        # Skeleton for masked_question Textfield to hold token_ids, mask, type_ids etc. after masking from ques_attn
        question_masked = {"tokens": {}}  # Our BERT token-indexer name is hardcoded to "tokens"
        input_keys = []   # "token_ids", "mask", "mask", "type_ids", "segment_concat_mask"
        for key in question["tokens"]:
            question_masked["tokens"][key] = []  # This would hold the input-tensors that will be concat before feeding
            input_keys.append(key)

        # For relevant actions, append the masked token_ids etc. in the question_masked TextField (from above)
        for (insidx, progidx, actionidx, question_attn) in insidxprogaction_qattn:
            # For each question-attention for an action, append a masked-question input
            question_attn_mask = (question_attn > question_attn_mask_threshold).long()
            for key in input_keys:
                input_tensor = question["tokens"][key][insidx, :].clone()
                if key not in ["type_ids", "segment_concat_mask"]:
                    input_tensor[1:-1] = question_attn_mask * input_tensor[1:-1]  # skip [CLS] and [SEP]
                question_masked["tokens"][key].append(input_tensor.unsqueeze(0))  # unsqueezing for concat later

        # Concatenate token_ids etc. into tensor for BERT
        for key in input_keys:
            # Shape: (num_relevant_actions, self.max_ques_len + 2)
            question_masked["tokens"][key] = torch.cat(question_masked["tokens"][key], dim=0)

        # Decontextualized question representation - (num_relevant_action, self.max_ques_len + 2, BERT_DIM)
        decont_q_reprs = text_field_embedder(question_masked)

        # To relevant action side_args_dict, add the weighted_question_vector
        group_idx = 0
        for (insidx, progidx, actionidx, question_attn) in insidxprogaction_qattn:
            question_attn_mask = (question_attn > question_attn_mask_threshold).float()
            sidearg_dict = batch_actionseq_sideargs[insidx][progidx][actionidx]
            decont_q = decont_q_reprs[group_idx, :, :][1:-1]    # Skip [CLS] and [SEP]
            # Mask the question-token representations and compute a sum representation
            weighted_question_repr = torch.sum(decont_q * question_attn_mask.unsqueeze(1), dim=0)
            sidearg_dict.update({"weighted_question_vector": weighted_question_repr})
            group_idx += 1


    def _add_decont_quespassage_repr(self,
                                     batch_action_seqs: List[List[List[str]]],
                                     batch_actionseq_sideargs: List[List[List[Dict]]],
                                     text_field_embedder: TextFieldEmbedder,
                                     question_passage: TextFieldTensors,
                                     max_ques_len: int,
                                     question_attn_mask_threshold: float = 0.1):
        """ Compute decontextualized question representation using question-passage as an input.

        Note:
            Shouldn't be used when computing a joint ques-passage encoding; it would render decontextualization useless

           For this to work with independent question & passage encoding the question TextField should be populated
           similarly to the question_passage TextField, i.e. the question TextField  should padded to max_question_len
           in the same manner as the initial half of the question_passage TextField.
           This is required since we're using the question-attention computed on the `question` TextField, in the
           question-subpart of the `question_passage` TextField here to perform decontextualization.
        """

        # List of (instance_idx, prog_idx, action_idx, question_attention) -- question-attention for relevant actions
        insidxprogaction_qattn = []

        # For all relevant actions, extract the (instance_idx, prog_idx, action_idx, question_attn)
        for instance_idx in range(len(batch_action_seqs)):
            for progidx, (action_seq, sideargs) in enumerate(zip(batch_action_seqs[instance_idx],
                                                                 batch_actionseq_sideargs[instance_idx])):
                for actionidx, (action_str, sidearg_dict) in enumerate(zip(action_seq, sideargs)):
                    if action_str in QReprModuleExecution.actions_w_qattn:
                        question_attention = sidearg_dict["question_attention"]
                        insidxprogaction_qattn.append((instance_idx, progidx, actionidx, question_attention))

        if len(insidxprogaction_qattn) == 0:
            # No relevant actions which need question-repr.
            return

        # Skeleton for masked_question_passage Textfield to hold token_ids, mask, type_ids etc. after masking from attn
        question_passage_masked = {"tokens": {}}  # Our BERT token-indexer name is hardcoded to "tokens"
        input_keys = []   # "token_ids", "mask", "mask", "type_ids", "segment_concat_mask"
        for key in question_passage["tokens"]:
            question_passage_masked["tokens"][key] = []  # This would hold the tensors that will be concat before input
            input_keys.append(key)

        # For relevant actions, append the masked token_ids etc. in the question_masked TextField (from above)
        for (insidx, progidx, actionidx, question_attn) in insidxprogaction_qattn:
            # For each question-attention for an action, append a masked-question input
            question_attn_mask = (question_attn > question_attn_mask_threshold).long()
            for key in input_keys:
                input_tensor = question_passage["tokens"][key][insidx, :].clone()
                if key not in ["type_ids", "segment_concat_mask"]:
                    input_tensor[1:max_ques_len + 1] = question_attn_mask * input_tensor[1:max_ques_len + 1]
                question_passage_masked["tokens"][key].append(input_tensor.unsqueeze(0))  # unsqueezing for concat later

        # Concatenate token_ids etc. into tensor for BERT
        for key in input_keys:
            # Shape: (num_relevant_actions, self.max_ques_len + 2)
            question_passage_masked["tokens"][key] = torch.cat(question_passage_masked["tokens"][key], dim=0)

        # Decontextualized question representation - (num_relevant_action, ques_passage_len, BERT_DIM)
        decont_q_reprs = text_field_embedder(question_passage_masked)

        # To relevant action side_args_dict, add the weighted_question_vector
        group_idx = 0
        for (insidx, progidx, actionidx, question_attn) in insidxprogaction_qattn:
            question_attn_mask = (question_attn > question_attn_mask_threshold).float()
            sidearg_dict = batch_actionseq_sideargs[insidx][progidx][actionidx]
            decont_q = decont_q_reprs[group_idx, 1:max_ques_len + 1, :]   # Skip [CLS] and keep max_ques_len tokens
            # Mask the question-token representations and compute a sum representation
            weighted_question_repr = torch.sum(decont_q * question_attn_mask.unsqueeze(1), dim=0)
            sidearg_dict.update({"weighted_question_vector": weighted_question_repr})
            group_idx += 1
