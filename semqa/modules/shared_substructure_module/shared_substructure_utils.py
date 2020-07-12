from typing import List, Tuple, Dict, Union, Optional

from enum import Enum

import torch
from allennlp.nn.util import move_to_device
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.modules.text_field_embedders import TextFieldEmbedder
import torch.nn.functional as F

from semqa.domain_languages.drop_language import DropLanguage, Output
from semqa.modules.qp_encodings import BertJointQPEncoding, QPEncoding
from semqa.utils.qdmr_utils import get_postorder_function_list, get_inorder_supervision_list, Node

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def compute_loss(
        device_id: int,
        max_ques_len: int,
        qp_encoder: QPEncoding,
        languages: List[DropLanguage],
        sharedsub_question_passage: TextFieldTensors,
        sharedsub_program_nodes: List[Union[Node, None]],
        sharedsub_function2actionidx_maps: List[Union[List[List[int]], None]],
        sharedsub_program_lisp: List[Union[List[str], None]],
        sharedsub_orig_program_lisp: List[Optional[str]],
        orig_sharedsub_postorder_node_idx: List[Union[List[Tuple[int, int]], None]],
        sharedsub_mask: torch.LongTensor,
        orig_action_seqs: List[List[str]],
        orig_program_outputs: List[List[Dict]],
):
    """Encode shared-substructure question, execute auxiliary-programs, and compute loss against original execution.

    Parameters:
    -----------
    sharedsub_question_passage: (batch_size, num_sharedsubs, ....) `TextFieldTensors`
        Each instance can contain multiple annotations, hence each of the keys in sharedsub_question_passage["tokens"],
        "token_ids", "mask", "type_ids", "segment_concat_mask", contains a wrapping dim.
        e.g. sharedsub_question_passage["tokens"]["token_ids"].size() == (bs, num_sharedsubs, passage_len)

    sharedsub_program_nodes: `List[Union[Dict, None]]`
        For each instance, a list of aux-program_supervision node-dict

    sharedsub_function2actionidx_maps: `List[Union[List[List[int]], None]]`
        For each instance, for each program, a list of idx mapping the function to the appropriate action in the
        action-seq corresponding the the program. This linearization is based on an in-order traversal of the program

    sharedsub_program_lisp: `List[List[str]]`
        For each instance, a list of aux-program lisps

    sharedsub_orig_program_lisp: `List[str]`
        For each instance, annotated lisp for the original question. Should be used to confirm that the correct
        program was decoded and we're using the outputs of the correct program.

    orig_sharedsub_postorder_node_idx: `List[List[Tuple[int, int]]]`
        For each instance, a list of tuples indicating the index of the node (in post-order traversal) in the
        original & shared program that should have the same output

    sharedsub_mask: (batch_size, num_sharedsubs) `torch.LongTensor`
        Mask for which shared-subs are padded

    orig_action_seqs: `List[List[str]]`
        Top decoded program (hopefully, the gold-annotated program) for the original question
    orig_module_outs: `List[List[Dict]]`
        Module debug info from execution of the top original-program
    """

    if not isinstance(qp_encoder, BertJointQPEncoding):
        raise NotImplementedError("qp_encoder needs to be a member of BertJointQPEncoding")

    BERT_KEYS = ["token_ids", "mask", "type_ids", "segment_concat_mask"]

    loss = 0.0
    batch_size, num_sharedsubs = sharedsub_mask.size()

    # Since each instance contains a variable (>= 0) number of annotations, we collect them in a flat representation
    #  for processing.
    group_idxs: List[int] = []      # batch_idx for the annotation. some might appear > 1 times, some might not at all
    flat_aux_question_passage: TextFieldTensors = {}
    flat_aux_program_lisps: List[str] = []
    flat_aux_program_nodes: List[Node] = []
    flat_aux_func2actionidx_mapping: List[List[int]] = []
    flat_orig_sharedsub_postorder_nodeidxs: List[Tuple[int, int]] = []

    flat_aux_question_passage["tokens"] = {}
    for key in BERT_KEYS:
        flat_aux_question_passage["tokens"][key] = []   # making list now which will be concat later

    for bidx in range(batch_size):
        for i in range(num_sharedsubs):
            if sharedsub_mask[bidx, i] == 0:
                continue
            group_idxs.append(bidx)
            for key in BERT_KEYS:
                flat_aux_question_passage["tokens"][key].append(
                    sharedsub_question_passage["tokens"][key][bidx, i, :].unsqueeze(0))
            flat_aux_program_lisps.append(sharedsub_program_lisp[bidx][i])
            flat_aux_program_nodes.append(sharedsub_program_nodes[bidx][i])
            flat_aux_func2actionidx_mapping.append(sharedsub_function2actionidx_maps[bidx][i])
            flat_orig_sharedsub_postorder_nodeidxs.append(orig_sharedsub_postorder_node_idx[bidx][i])

    if not group_idxs:
        return loss

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

    for i, batch_idx in enumerate(group_idxs):
        orig_decoded_action_seq: List[str] = orig_action_seqs[batch_idx]
        orig_decoded_prog_lisp: str = languages[i].action_sequence_to_logical_form(orig_decoded_action_seq)
        # this is the original lisp against which the shared-sub example is written
        orig_annotated_prog_lisp = sharedsub_orig_program_lisp[batch_idx]
        if orig_decoded_prog_lisp != orig_annotated_prog_lisp:
            logger.warning(f"Original-ques annotated lisp is not the same as decoded:\n"
                           f"annotated: {orig_annotated_prog_lisp}  "
                           f"decoded: {orig_decoded_prog_lisp}")

        program_node: Node = flat_aux_program_nodes[i]
        program_lisp = flat_aux_program_lisps[i]

        function2actionidx_map: List[int] = flat_aux_func2actionidx_mapping[i]
        language = languages[batch_idx]
        action_seq: List[str] = language.logical_form_to_action_sequence(program_lisp)
        prog_sideargs: List[Dict] = [{} for _ in action_seq]

        # These are the action-idxs which produce terminal-node functions; gold_function2actionidx_maps is
        # packed so that as multiple gold-programs are passed. we assume singe gold-program here

        inorder_supervision_dicts: List[Dict] = get_inorder_supervision_list(program_node)
        assert len(function2actionidx_map) == len(inorder_supervision_dicts), "each func. should have a supdict"
        # Append the appropriate pred_sideargs idxs with the supervision-dict
        for action_idx, supervision_dict in zip(function2actionidx_map, inorder_supervision_dicts):
            prog_sideargs[action_idx].update(supervision_dict)

        for sidearg_dict in prog_sideargs:
            if "question_attention_supervision" in sidearg_dict:
                # For shared-substructure aux programs, the supervised question-attention will be used as the predicted
                # question attention
                gold_attn = sidearg_dict["question_attention_supervision"]
                sidearg_dict["question_attention"] = sidearg_dict["question_attention_supervision"]
                gold_attn_tensor: torch.FloatTensor = move_to_device(
                    torch.FloatTensor(gold_attn), cuda_device=device_id)

                gold_attn_len = len(gold_attn)
                q_encoding = encoded_question[i, 0:gold_attn_len, :]
                weighted_question_vector = torch.sum(
                    q_encoding * gold_attn_tensor.unsqueeze(1), dim=0)
                sidearg_dict["weighted_question_vector"] = weighted_question_vector

        language.modules_debug_info.append([])
        actionseq_denotation = language.execute_action_sequence(action_seq, prog_sideargs)

        sharedsub_prog_outputs: List[Dict] = language.modules_debug_info[-1]
        orig_prog_outputs: List[Dict] = orig_program_outputs[batch_idx]
        orig_sharednode_idx, sharedsub_sharednode_idx = flat_orig_sharedsub_postorder_nodeidxs[i]

        orig_module_outputs: Dict[str, List[Output]] = orig_prog_outputs[orig_sharednode_idx]
        sharedsub_module_outputs: Dict[str, List[Output]] = sharedsub_prog_outputs[sharedsub_sharednode_idx]

        orig_output_type, orig_output_tensor = get_module_output(orig_module_outputs)
        sharedsub_output_type, sharedsub_output_tensor = get_module_output(sharedsub_module_outputs)

        if orig_output_type != sharedsub_output_type:
            logger.warning(f"Orig output type ({orig_output_type}) is not the same as "
                           f"sharedsub output type ({sharedsub_output_type})")
        # Ideally we would perform a masked_log, but due to clamping masked values are 1e-20
        loss += (F.kl_div(orig_output_tensor.log(), sharedsub_output_tensor, reduction="mean") +
                 F.kl_div(sharedsub_output_tensor.log(), orig_output_tensor, reduction="mean"))

    return loss

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

