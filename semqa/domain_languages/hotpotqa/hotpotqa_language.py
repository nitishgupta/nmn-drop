from typing import Callable, List, Dict
import logging

import torch
import torch.nn.functional
from torch import Tensor


from allennlp.common.registrable import Registrable
import allennlp.nn.util as allenutil
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
from allennlp.semparse import (DomainLanguage, ExecutionError, ParsingError,
                               predicate, predicate_with_side_args, util)
from allennlp.semparse.domain_languages.domain_language import PredicateType

import datasets.hotpotqa.utils.constants as hpcons


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ExecutorParameters(torch.nn.Module, Registrable):
    """
    Global parameters for execution. Executor objects are made for each instance, where as these will be shared across.
    """
    def __init__(self,
                 ques_encoder: Seq2SeqEncoder,
                 context_embedder: TextFieldEmbedder,
                 context_encoder: Seq2SeqEncoder):
        super(ExecutorParameters, self).__init__()
        self._ques_encoder = ques_encoder
        self._context_embedder = context_embedder
        self._context_encoder = context_encoder
        self._span_extractor = EndpointSpanExtractor(input_dim=self._context_encoder.get_output_dim())


    def _encode_contexts(self, contexts: Dict[str, torch.LongTensor]) -> torch.FloatTensor:
        """ Encode the contexts for each instance using the context_encoder RNN.

        Params:
        -------
        contexts: ``Dict[str, torch.LongTensor]``
            Since there are multiple contexts per instance, the contexts tensor are wrapped as (B, C, T)
            where C is the number of contexts

        Returns:
        --------
        contexts_encoded: ``torch.FloatTensor``
            Tensor of shape (B, C, T, D) after encoding all the contexts for each instance in the batch
        """

        # Shape: (B, C, T, W_d)
        embedded_contexts = self._context_embedder(contexts)
        embcontext_size = embedded_contexts.size()

        # Shape: (B, C, T)
        # Since multiple contexts per instance, give num_wrapping_dims
        contexts_mask = allenutil.get_text_field_mask(contexts, num_wrapping_dims=1).float()
        conmask_size = contexts_mask.size()

        (embedded_contexts_flat, contexts_mask_flat) = (
        embedded_contexts.view(-1, embcontext_size[2], embcontext_size[3]),
        contexts_mask.view(-1, conmask_size[2]))

        # Shape: (B*C, T, D)
        contexts_encoded_flat = self._context_encoder(embedded_contexts_flat, contexts_mask_flat)
        conenc_size = contexts_encoded_flat.size()
        # View such that get B, C, T from embedded context, and D from encoded contexts
        # Shape: (B, C, T, D)
        contexts_encoded = contexts_encoded_flat.view(*embcontext_size[0:3], conenc_size[-1])

        return contexts_encoded, contexts_mask


class Bool():
    def __init__(self, value: Tensor):
        self._value = self.clamp(value)
        self._bool_val = (self._value >= 0.5).float()

    def clamp(self, value: torch.Tensor):
        new_val = value.clamp(min=1e-3, max=1.0 - 1e-3)
        return new_val


class Bool1():
    def __init__(self, value: Tensor):
        self._value = value
        self.clamp()
        self._bool_val = (self._value >= 0.5).float()

    def clamp(self):
        self._value = self._value.clamp(min=1e-1, max=1.0 - 1e-1)


class Qstr(str):
    pass


class Qent(str):
    pass


class HotpotQALanguage(DomainLanguage):
    def __init__(self, qstr_qent_spans: List[str]):
        super().__init__(start_types={Bool})

        self._add_constants(qstr_qent_spans=qstr_qent_spans)

        # These are mappings from Type predicates to the name of the type in the preprocessed data
        # Useful for (1) Finding starting rules that result in a particular type. (2) Finding the type of a logical prog.
        self.LANGTYPE_TO_TYPENAME_MAP = {
            PredicateType.get_type(Bool).name: hpcons.BOOL_TYPE
        }

        self.TYPENAME_TO_LANGTYPE_MAP = {
            hpcons.BOOL_TYPE: PredicateType.get_type(Bool).name
        }

        self._execution_parameters = None

        self.qstr2repr = None
        self.ne_ent_mens = None
        self.contexts = None
        self.contexts_mask = None
        self.q_nemenspan2entidx = None

        self.ques_embedded = None
        self.ques_mask = None
        self.contexts = None
        self.ne_ent_mens = None
        self.num_ent_mens = None
        self.date_ent_mens = None
        self.q_qstr2idx = None
        self.q_qstr_spans = None
        self.q_nemens2groundingidx = None
        self.q_nemens_grounding = None
        self.q_nemenspan2entidx = None


    def _add_constants(self, qstr_qent_spans: List[str]):
        for span in qstr_qent_spans:
            if span.startswith(hpcons.QSTR_PREFIX):
                self.add_constant(name=span, value=span, type_=Qstr)
            elif span.startswith(hpcons.QENT_PREFIX):
                # Question NE men span
                self.add_constant(name=span, value=span, type_=Qent)


    def typeobj_to_typename(self, obj):
        ''' For a object of a return-type of the language, return the name of type used in the dataset.

        The obj input here should be a non-terminal type of the language defined, ideally the output of an execution
        '''
        # This is the name of the equivalent BasicType
        class_name =  PredicateType.get_type(obj.__class__).name
        if class_name in self.LANGTYPE_TO_TYPENAME_MAP:
            return self.LANGTYPE_TO_TYPENAME_MAP[class_name]
        else:
            raise ExecutionError(message="TypeClass to Type mapping not found! ClassName:{}".format(class_name))

    def typename_to_langtypename(self, typename):
        ''' For a given name of a type used in the dataset, return the mapped name of the type (BasicType) in the lang.

        The typename entered here should have a mapping to a non-terminal type of the language defined.
        '''
        # This is the name of the type from the dataset
        if typename in self.TYPENAME_TO_LANGTYPE_MAP:
            return self.TYPENAME_TO_LANGTYPE_MAP[typename]
        else:
            raise ExecutionError(message="Mapping from type name to Type not found! TypeName:{}".format(typename))


    def set_arguments(self, **kwargs):
        self.ques_embedded = kwargs["ques_embedded"]
        self.ques_mask = kwargs["ques_mask"]
        self.contexts = kwargs["contexts"]
        self.contexts_mask = kwargs["contexts_mask"]
        self.ne_ent_mens = kwargs["ne_ent_mens"]
        self.num_ent_mens = kwargs["num_ent_mens"]
        self.date_ent_mens = kwargs["date_ent_mens"]
        self.q_qstr2idx= kwargs["q_qstr2idx"]
        self.q_qstr_spans = kwargs["q_qstr_spans"]
        self.q_nemens2groundingidx = kwargs["q_nemens2groundingidx"]
        self.q_nemens_grounding = kwargs["q_nemens_grounding"]
        self.q_nemenspan2entidx = kwargs["q_nemenspan2entidx"]


    def set_execution_parameters(self, execution_parameters: ExecutorParameters):
        self._execution_parameters: ExecutorParameters = execution_parameters


    def preprocess_arguments(self):
        """ Preprocessing arguments to make tensors that can be reused across logical forms during execution.

        For example, question span representations will be precomputed and stored here to reuse when executing different
        logical forms.
        """

        # For all QSTR, computing repr by running ques_encoder, and concatenating repr of end-points.
        self.qstr2repr = {}
        for qstr, qstr_idx in self.q_qstr2idx.items():
            # Shape: [2] denoting the span in the question
            qstr_span = self.q_qstr_spans[qstr_idx]
            # Shape: (QSTR_len, emb_dim)
            qstr_embedded = self.ques_embedded[qstr_span[0]:qstr_span[1] + 1]
            qstr_mask = self.ques_mask[qstr_span[0]:qstr_span[1] + 1]
            # [1, QSTR_len, emb_dim]
            qstr_encoded_ex = self._execution_parameters._ques_encoder(qstr_embedded.unsqueeze(0), qstr_mask.unsqueeze(0))
            # Shape: [QSTR_len, emb_dim]
            qstr_encoded = qstr_encoded_ex.squeeze(0)
            # Concatenating first and last time step of encoded qstr
            # Shape: (2 * Qd)
            qstr_repr = torch.cat([qstr_encoded[0].unsqueeze(0), qstr_encoded[-1].unsqueeze(0)], 1).squeeze(0)
            self.qstr2repr[qstr] = qstr_repr


    @predicate
    def bool_qent_qstr(self, qent: Qent, qstr: Qstr) -> Bool1:

        # Get Q_ent string, and Q_str and map to boolean.

        entity_grounding_idx = self.q_nemenspan2entidx[qent]

        # Shape: (2*D)
        qstr_repr = self.qstr2repr[qstr]

        # qstr_span = self.q_qstr_spans[self.q_qstr2idx[qstr]]
        # # [QSTR_len, emb_dim]
        # qstr_embedded = self.ques_embedded[qstr_span[0]:qstr_span[1]+1]
        # qstr_mask = self.ques_mask[qstr_span[0]:qstr_span[1]+1]
        # # [1, QSTR_len, emb_dim]
        # qstr_encoded_ex = self._executor_parameters._ques_encoder(qstr_embedded.unsqueeze(0), qstr_mask.unsqueeze(0))
        # # Shape: [QSTR_len, emb_dim]
        # qstr_encoded = qstr_encoded_ex.squeeze(0)
        # # Concatenating first and last time step of encoded qstr
        # # Shape: (2 * Qd)
        # qstr_repr = torch.cat([qstr_encoded[0].unsqueeze(0), qstr_encoded[-1].unsqueeze(0)], 1).squeeze(0)

        # Shape: (C, M, 2)
        qent_mens = self.ne_ent_mens[entity_grounding_idx]
        qent_mens_mask = (qent_mens[..., 0] >= 0).long()
        # Shape: (C, M, 2*D)
        # print("Exectution debug")
        # print(self.contexts.size())
        # print(qent_mens)
        # print(qent_mens_mask)
        qent_men_repr = self._execution_parameters._span_extractor(self.contexts, qent_mens,
                                                                   self.contexts_mask, qent_mens_mask)
        qstr_repr_ex = qstr_repr.unsqueeze(0).unsqueeze(1)
        scores = torch.sum(qstr_repr_ex * qent_men_repr, 2)
        probs = torch.sigmoid(scores) * qent_mens_mask.float()

        boolean_prob = torch.max(probs) #.unsqueeze(0)

        return Bool1(value=boolean_prob)


    @predicate
    def bool_and(self, bool1: Bool1, bool2: Bool1) -> Bool:
        """ AND operation between two booleans """
        return_val = bool1._value * bool2._value
        # torch.sigmoid(10.0*bool1._value + 10.0*bool2._value - 15.0)
        return Bool(value=return_val)
        # return Bool(value=bool1._value * bool2._value)

    @predicate
    def bool_or(self, bool1: Bool1, bool2: Bool1) -> Bool:
        """ OR operation between two booleans """
        return_val = torch.max(torch.cat([bool1._value.unsqueeze(0), bool2._value.unsqueeze(0)], 0))
        return Bool(value=return_val)


if __name__=="__main__":

    b = Bool(value=torch.FloatTensor([0.65]))
    print(b._value)
    print(b._bool_val)
    print(PredicateType.get_type(b.__class__).name)
    print(PredicateType.get_type(Bool).name)

    # language = HotpotQALanguage(qstr_qent_spans=["QSTR:Hi", "QENT:Nitish"])
    # all_prods = language.all_possible_productions()
    #
    # print("All prods:\n{}\n".format(all_prods))
    #
    # nonterm_prods = language.get_nonterminal_productions()
    # print("Non terminal prods:\n{}\n".format(nonterm_prods))
    #
    # functions = language._functions
    # print("Functions:\n{}\n".format(functions))
    #
    # function_types = language._function_types
    # print("Function Types:\n{}\n".format(function_types))
    #
    # logical_form = "(bool_qent_qstr QSTR:Hi QENT:Nitish)"
    # action_seq = language.logical_form_to_action_sequence(logical_form=logical_form)
    # print(action_seq)




