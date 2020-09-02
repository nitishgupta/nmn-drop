from typing import List, Tuple, Dict, Union, Optional

from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.nn.util import move_to_device, masked_softmax
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp_semparse.fields.production_rule_field import ProductionRule
from allennlp_semparse.state_machines import ConstrainedBeamSearch

from semqa.domain_languages.drop_execution_parameters import ExecutorParameters
from semqa.domain_languages.drop_language import DropLanguage, Output
from semqa.modules.qp_encodings import BertJointQPEncoding, QPEncoding
from semqa.modules.symbolic.utils import compute_token_symbol_alignments
from semqa.utils.qdmr_utils import get_postorder_function_list, get_inorder_supervision_list, Node
from semqa.modules.qrepr_module.qrepr_to_module import QReprModuleExecution

from semqa.models.utils import semparse_utils
# from semqa.models.squad_parser_bert import DROPParserBERT

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def compute_loss(
        device_id: int,
        max_ques_len: int,
        executor_parameters: ExecutorParameters,
        passageidx2dateidx: torch.LongTensor,
        passageidx2numberidx: torch.LongTensor,
        qp_encoder: QPEncoding,
        languages: List[DropLanguage],
        actions: List[List[ProductionRule]],
        paired_example_mask: torch.LongTensor,
        paired_question_passage: TextFieldTensors,
        paired_program_nodes: List[List[Optional[Node]]],
        paired_function2actionidx_maps: List[List[Optional[List[int]]]],
        paired_program_lisp: List[List[Optional[str]]],
        paired_orig_program_lisp: List[Optional[str]],
        orig_paired_postorder_sharednode_idx: List[Union[List[Tuple[int, int]], None]],
        paired_action_seqs,
        paired_passage_span_answer,
        paired_passage_span_answer_mask,
        paired_passage_numbers_answers,
        paired_count_answers,
        orig_action_seqs: List[List[str]],
        year_differences_mat: List[np.array],
        orig_program_outputs: List[List[Dict]],
        metadata: List,
        bert_nmn_model, #: DROPParserBERT,
):
    """Encode shared-substructure question, execute auxiliary-programs, and compute loss against original execution.

    Parameters:
    -----------
    paired_example_mask: (batch_size, (max)num_paired_examples) `torch.LongTensor`
        Mask for which paired_examples are padded

    paired_question_passage: (batch_size, num_sharedsubs, ....) `TextFieldTensors`
        Each instance can contain multiple annotations, hence each of the keys in sharedsub_question_passage["tokens"],
        "token_ids", "mask", "type_ids", "segment_concat_mask", contains a wrapping dim.
        e.g. sharedsub_question_passage["tokens"]["token_ids"].size() == (bs, num_sharedsubs, passage_len)

    paired_program_nodes: `List[List[Optional[Node]]]`
        For each instance, a list of aux-program_supervision node-dict

    paired_function2actionidx_maps: `List[List[Optional[List[int]]]]`
        For each instance, for each program, mapping the function-node in the program-tree to the appropriate action
        in the corresponding action-seq. The function idx is its index in an inorder linearization of the program tree.
        Used to map node-level supervision to the correct action's side-arg

    paired_program_lisp: `List[List[str]]`
        For each instance, a list of aux-program lisps

    paired_orig_program_lisp: `List[str]`
        For each instance, annotated lisp for the original question. Should be used to confirm that the correct
        program was decoded and we're using the outputs of the correct program.

    orig_sharedsub_postorder_node_idx: `List[List[Tuple[int, int]]]`
        For each instance, a list of tuples indicating the index of the node (in post-order traversal) in the
        original & shared program that should have the same output

    orig_action_seqs: `List[List[str]]`
        Top decoded program (hopefully, the gold-annotated program) for the original question
    orig_module_outs: `List[List[Dict]]`
        Module debug info from execution of the top original-program
    """

    if not isinstance(qp_encoder, BertJointQPEncoding):
        raise NotImplementedError("qp_encoder needs to be a member of BertJointQPEncoding")

    BERT_KEYS = ["token_ids", "mask", "type_ids", "segment_concat_mask"]

    paired_loss, paired_denotation_loss = 0.0, 0.0
    batch_size, num_sharedsubs = paired_example_mask.size()

    # Since each instance contains a variable (>= 0) number of annotations, we collect them in a flat representation
    #  for processing.
    group_idxs: List[int] = []      # batch_idx for the annotation. some might appear > 1 times, some might not at all
    flat_aux_question_passage: TextFieldTensors = {}
    flat_aux_program_lisps: List[str] = []
    flat_aux_program_nodes: List[Node] = []
    flat_aux_func2actionidx_mapping: List[List[int]] = []
    flat_orig_sharedsub_postorder_nodeidxs: List[Tuple[int, int]] = []
    flat_aux_action_seqs = []
    flat_aux_passage_span_answer = []
    flat_aux_passage_span_answer_mask = []
    flat_aux_passage_number_answer = []
    flat_aux_count_answer = []

    # Slice the original p2date and p2num indices and concat to make mapping for auxiliary
    aux_passageidx2dateidx_list = []
    aux_passageidx2numberidx_list = []

    flat_aux_question_passage["tokens"] = {}
    for key in BERT_KEYS:
        flat_aux_question_passage["tokens"][key] = []   # making list now which will be concat later

    for bidx in range(batch_size):
        for i in range(num_sharedsubs):
            if paired_example_mask[bidx, i] == 0:
                continue
            group_idxs.append(bidx)
            for key in BERT_KEYS:
                flat_aux_question_passage["tokens"][key].append(
                    paired_question_passage["tokens"][key][bidx, i, :].unsqueeze(0))
            flat_aux_program_lisps.append(paired_program_lisp[bidx][i])
            flat_aux_program_nodes.append(paired_program_nodes[bidx][i])
            flat_aux_func2actionidx_mapping.append(paired_function2actionidx_maps[bidx][i])
            flat_orig_sharedsub_postorder_nodeidxs.append(orig_paired_postorder_sharednode_idx[bidx][i])
            flat_aux_action_seqs.append(paired_action_seqs[bidx][i])
            flat_aux_passage_span_answer.append(paired_passage_span_answer[bidx, i, :, :].unsqueeze(0))
            flat_aux_passage_span_answer_mask.append(paired_passage_span_answer_mask[bidx, i, :].unsqueeze(0))
            flat_aux_passage_number_answer.append(paired_passage_numbers_answers[bidx][i])
            flat_aux_count_answer.append(paired_count_answers[bidx][i])
            aux_passageidx2dateidx_list.append(passageidx2dateidx[bidx].unsqueeze(0))
            aux_passageidx2numberidx_list.append(passageidx2numberidx[bidx].unsqueeze(0))

    if not group_idxs:
        return paired_loss, paired_denotation_loss

    for key in BERT_KEYS:
        flat_aux_question_passage["tokens"][key] = torch.cat(flat_aux_question_passage["tokens"][key], dim=0)

    flat_aux_qp_encoding = qp_encoder.get_representation(question_passage=flat_aux_question_passage,
                                                         max_ques_len=max_ques_len)

    question_token_idxs = flat_aux_qp_encoding["question_token_idxs"]
    passage_token_idxs = flat_aux_qp_encoding["passage_token_idxs"]
    question_mask = flat_aux_qp_encoding["question_mask"]
    passage_mask = flat_aux_qp_encoding["passage_mask"]
    encoded_question = flat_aux_qp_encoding["encoded_question"]
    encoded_passage = flat_aux_qp_encoding["encoded_passage"]
    bert_pooled_out = flat_aux_qp_encoding["pooled_encoding"]
    question_encoded_final_state = bert_pooled_out

    aux_passageidx2dateidx = torch.cat(aux_passageidx2dateidx_list, dim=0)
    aux_passageidx2numberidx = torch.cat(aux_passageidx2numberidx_list, dim=0)

    passage_len = passage_mask.size()[1]
    aux_passageidx2dateidx = aux_passageidx2dateidx[:, 0:passage_len]
    aux_passageidx2numberidx = aux_passageidx2numberidx[:, 0:passage_len]

    # (num_paired_examples, num_spans, passage_len)
    flat_aux_passage_span_answer = torch.cat(flat_aux_passage_span_answer, dim=0)[:, :, 0:passage_len]
    flat_aux_passage_span_answer_mask = torch.cat(flat_aux_passage_span_answer_mask, dim=0)

    aux_languages = []
    aux_actions: List[List[ProductionRule]] = []
    for i, batch_idx in enumerate(group_idxs):
        language = languages[batch_idx]
        aux_language = DropLanguage(
                    encoded_passage=encoded_passage[i],
                    modeled_passage=encoded_passage[i],
                    passage_mask=passage_mask[i],
                    passage_sentence_boundaries=language.passage_sentence_boundaries,
                    passage_tokenidx2dateidx=aux_passageidx2dateidx[i],
                    passage_date_values=language.passage_date_values,
                    passage_tokenidx2numidx=aux_passageidx2numberidx[i],
                    passage_num_values=language.passage_num_values,
                    composed_numbers=language.composed_numbers,
                    passage_number_sortedtokenidxs=language.passage_number_sortedtokenidxs,
                    add_num_combination_indices=language.add_num_combination_indices,
                    sub_num_combination_indices=language.sub_num_combination_indices,
                    year_differences=language.year_differences,
                    year_differences_mat=year_differences_mat[batch_idx],
                    count_num_values=language.count_num_values,
                    parameters=language.parameters,
                    start_types=None,  # batch_start_types[i],
                    device_id=device_id,
                    debug=language._debug,
                    metadata=metadata[batch_idx]
        )
        aux_languages.append(aux_language)
        aux_actions.append(actions[batch_idx])

    (initial_state, _, _) = bert_nmn_model.getInitialDecoderState(
        languages=aux_languages,
        actions=aux_actions,
        encoded_question=encoded_question,
        question_mask=question_mask,
        question_encoded_final_state=question_encoded_final_state,
        question_encoded_aslist=[encoded_question[i] for i in range(len(group_idxs))],
        question_mask_aslist=[question_mask[i] for i in range(len(group_idxs))],
        batch_size=len(group_idxs))

    aux_actionseq_idxs, aux_actionseq_masks = zip(*flat_aux_action_seqs)
    aux_actionseq_idxs = list(aux_actionseq_idxs)
    aux_actionseq_masks = list(aux_actionseq_masks)

    constrained_search = ConstrainedBeamSearch(
        bert_nmn_model._beam_size,
        allowed_sequences=aux_actionseq_idxs,
        allowed_sequence_mask=aux_actionseq_masks,
    )

    final_states = constrained_search.search(
        initial_state=initial_state, transition_function=bert_nmn_model._decoder_step
    )

    (
        batch_actionidxs,
        batch_actionseqs,
        batch_actionseq_logprobs,
        batch_actionseq_sideargs,
    ) = semparse_utils._convert_finalstates_to_actions(
        best_final_states=final_states, possible_actions=aux_actions, batch_size=len(group_idxs)
    )

    bert_nmn_model.qrepr_module_exec.add_weighted_question_vector_to_sideargs(
        batch_action_seqs=batch_actionseqs,
        batch_actionseq_sideargs=batch_actionseq_sideargs,
        text_field_embedder=bert_nmn_model._text_field_embedder,
        question_encoding=encoded_question,
        question_mask=question_mask,
        question=None,
        question_passage=None,
        max_ques_len=bert_nmn_model.max_ques_len,
        device_id=bert_nmn_model.device_id,
        question_attn_mask_threshold=0.1)

    # List[List[Any]], List[List[str]]: Denotations and their types for all instances
    flat_aux_denotations, flat_aux_denotation_types = bert_nmn_model._get_denotations(
        batch_actionseqs, aux_languages, batch_actionseq_sideargs
    )

    for i, batch_idx in enumerate(group_idxs):
        # Denotation loss
        # Final denotation loss -- only works for PassageSpanAnswer and BIO-tagging
        denotation = flat_aux_denotations[i][0]  # zero since there is a single paired example in the flat notation
        denotation_type = flat_aux_denotation_types[i][0]

        denotation_loss = move_to_device(torch.tensor(0.0), bert_nmn_model.device_id).float()
        prog_log_prob = batch_actionseq_logprobs[i][0]
        if denotation_type == "PassageSpanAnswer":
            span_answer_loss_inputs = {"passage_span_answer": flat_aux_passage_span_answer[i],
                                       "passage_span_answer_mask": flat_aux_passage_span_answer_mask[i],
                                       "log_probs": denotation.bio_logprobs,
                                       "passage_mask": passage_mask[i, :]}

            log_likelihood = bert_nmn_model.span_answer.gold_log_marginal_likelihood(**span_answer_loss_inputs)
            denotation_loss = -1.0 * (log_likelihood + prog_log_prob)

        elif denotation_type == "PassageNumber":
            gold_passage_number_answer = flat_aux_passage_number_answer[i]
            pred_passagenumber_dist = denotation._value
            pred_passagenumber_logprobs = torch.log(pred_passagenumber_dist + 1e-40)
            gold_passagenum_dist = move_to_device(
                torch.FloatTensor(gold_passage_number_answer), cuda_device=bert_nmn_model.device_id)
            log_likelihood = torch.sum(pred_passagenumber_logprobs * gold_passagenum_dist)
            denotation_loss = -1.0 * (log_likelihood + prog_log_prob)
        elif denotation_type == "CountNumber":
            gold_count_answer = flat_aux_count_answer[i]
            pred_count_dist = denotation._value
            pred_count_logprobs = torch.log(pred_count_dist + 1e-40)
            gold_count_dist = move_to_device(
                torch.FloatTensor(gold_count_answer), cuda_device=bert_nmn_model.device_id)
            log_likelihood = torch.sum(pred_count_logprobs * gold_count_dist)
            denotation_loss = -1.0 * (log_likelihood + prog_log_prob)
        else:
            raise NotImplementedError

        paired_denotation_loss += denotation_loss

        orig_decoded_action_seq: List[str] = orig_action_seqs[batch_idx]
        orig_decoded_prog_lisp: str = languages[batch_idx].action_sequence_to_logical_form(orig_decoded_action_seq)
        # this is the original lisp against which the shared-sub example is written
        orig_annotated_prog_lisp = paired_orig_program_lisp[batch_idx]
        if orig_decoded_prog_lisp != orig_annotated_prog_lisp:
            logger.warning(f"Original-ques annotated lisp is not the same as decoded:\n"
                           f"annotated: {orig_annotated_prog_lisp}  "
                           f"decoded: {orig_decoded_prog_lisp}")

        program_node: Node = flat_aux_program_nodes[i]
        program_lisp = flat_aux_program_lisps[i]

        """
        function2actionidx_map: List[int] = flat_aux_func2actionidx_mapping[i]
        language = aux_languages[i]
        # language = languages[batch_idx]
        action_seq: List[str] = language.logical_form_to_action_sequence(program_lisp)
        prog_sideargs: List[Dict] = [{} for _ in action_seq]

        inorder_supervision_dicts: List[Dict] = get_inorder_supervision_list(program_node)
        assert len(function2actionidx_map) == len(inorder_supervision_dicts), "each func. should have a supdict"
        # Update the side-arg with the appropriate supervision-dict
        for action_idx, supervision_dict in zip(function2actionidx_map, inorder_supervision_dicts):
            prog_sideargs[action_idx].update(supervision_dict)

        for action, sidearg_dict in zip(action_seq, prog_sideargs):
            # TODO: not all relevant actions would have the needed question-attention-supervision; in that case we
            #  currently just use an all-0s attention assuming that the loss will only be computed on predicates below
            #  such actions in the tree. E.g. In select_num(select), say select_num doesn't have the reqd. supervision,
            #  but the loss would only be computed on select.
            if action in QReprModuleExecution.actions_w_qattn:
                if "question_attention_supervision" not in sidearg_dict:
                    qlen = encoded_question.size(1)
                    gold_attn_tensor = move_to_device(torch.zeros(qlen), cuda_device=device_id)
                    gold_attn_len = qlen
                    sidearg_dict["question_attention"] = gold_attn_tensor
                else:
                    # For shared-substructure aux programs, the supervised question-attention will be used as the predicted
                    # question attention for weighted_question_vector
                    gold_attn = sidearg_dict["question_attention_supervision"]
                    sidearg_dict["question_attention"] = sidearg_dict["question_attention_supervision"]
                    gold_attn_tensor: torch.FloatTensor = move_to_device(
                        torch.FloatTensor(gold_attn), cuda_device=device_id)
                    gold_attn_tensor = F.softmax(gold_attn_tensor, dim=0)
                    gold_attn_len = len(gold_attn)
                q_encoding = encoded_question[i, 0:gold_attn_len, :]
                weighted_question_vector = torch.sum(
                    q_encoding * gold_attn_tensor.unsqueeze(1), dim=0)
                sidearg_dict["weighted_question_vector"] = weighted_question_vector

        language.modules_debug_info.append([])

        try:
            actionseq_denotation = language.execute_action_sequence(action_seq, prog_sideargs)
        except:
            print(action_seq)
            print(program_lisp)
            continue
        """
        aux_language = aux_languages[i]
        sharedsub_prog_outputs: List[Dict] = aux_language.modules_debug_info[-1]
        orig_prog_outputs: List[Dict] = orig_program_outputs[batch_idx]
        orig_sharednode_idx, sharedsub_sharednode_idx = flat_orig_sharedsub_postorder_nodeidxs[i]

        orig_module_outputs: Dict[str, List[Output]] = orig_prog_outputs[orig_sharednode_idx]
        sharedsub_module_outputs: Dict[str, List[Output]] = sharedsub_prog_outputs[sharedsub_sharednode_idx]

        # Type of output (passage_attn) and value tensor
        orig_output_type, orig_output_tensor = get_module_output(orig_module_outputs)
        sharedsub_output_type, sharedsub_output_tensor = get_module_output(sharedsub_module_outputs)

        if orig_output_type != sharedsub_output_type:
            logger.warning(f"Orig output type ({orig_output_type}) is not the same as "
                           f"sharedsub output type ({sharedsub_output_type})")
            continue

        if orig_output_tensor.size() != sharedsub_output_tensor.size():
            # Needed when sharedsub passage get's less padding than the original
            length = sharedsub_output_tensor.size()[0]
            orig_output_tensor = orig_output_tensor[:length]

        # Ideally we would perform a masked_log, but due to clamping masked values are 1e-20
        paired_loss += (F.kl_div(orig_output_tensor.log(), sharedsub_output_tensor, reduction="mean") +
                 F.kl_div(sharedsub_output_tensor.log(), orig_output_tensor, reduction="mean"))

    return paired_loss, paired_denotation_loss


def compute_aux_token_symbol_alignments(modeled_passage, passage_mask, executor_parameters,
                                        passageidx2dateidx, passageidx2numberidx):
    # Passage Token - Date Alignment
    # Shape: (batch_size, passage_length, passage_length)
    passage_passage_token2date_alignment = compute_token_symbol_alignments(
        modeled_passage=modeled_passage,
        passage_mask=passage_mask,
        passageidx2symbolidx=passageidx2dateidx,
        passage_to_symbol_attention_params=executor_parameters.passage_to_date_attention
    )

    passage_passage_token2startdate_alignment = compute_token_symbol_alignments(
        modeled_passage=modeled_passage,
        passage_mask=passage_mask,
        passageidx2symbolidx=passageidx2dateidx,
        passage_to_symbol_attention_params=executor_parameters.passage_to_start_date_attention
    )

    passage_passage_token2enddate_alignment = compute_token_symbol_alignments(
        modeled_passage=modeled_passage,
        passage_mask=passage_mask,
        passageidx2symbolidx=passageidx2dateidx,
        passage_to_symbol_attention_params=executor_parameters.passage_to_end_date_attention
    )
    # Passage Token - Num Alignment
    passage_passage_token2num_alignment = compute_token_symbol_alignments(
        modeled_passage=modeled_passage,
        passage_mask=passage_mask,
        passageidx2symbolidx=passageidx2numberidx,
        passage_to_symbol_attention_params=executor_parameters.passage_to_num_attention
    )

    return (passage_passage_token2date_alignment, passage_passage_token2startdate_alignment,
            passage_passage_token2enddate_alignment, passage_passage_token2num_alignment)

def get_module_output(module_outputs: Dict[str, List[Output]]) -> Tuple[str, torch.Tensor]:
    """Get the actual module output to compute loss against given all debug-outputs from a module.

    module_outputs: `Dict[str, List[Output]]`
        Single entry where key: module_name and list of Outputs are the different outputs
    """
    module_name: str = list(module_outputs.keys())[0]
    if module_name == "select_passage":
        for output in module_outputs[module_name]:
            if output.label == "passage_attn":
                return "passage_attn", output.values
    else:
        raise NotImplementedError

