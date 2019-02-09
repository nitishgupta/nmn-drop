from typing import List
import logging

import torch
import torch.nn.functional
from torch import Tensor

import allennlp.nn.util as allenutil
from allennlp.semparse import (DomainLanguage, ExecutionError, predicate, predicate_with_side_args)
from allennlp.semparse.domain_languages.domain_language import PredicateType

import datasets.hotpotqa.utils.constants as hpcons
from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        self._value = self._value.clamp(min=1e-3, max=1.0 - 1e-3)


class Qstr():
    def __init__(self, attention):
        self._value = attention


class Qent():
    def __init__(self, attention):
        self._value = attention


class HotpotQALanguage(DomainLanguage):
    def __init__(self, qstr_qent_spans: List[str]):
        super().__init__(start_types={Bool})
        # self._add_constants(qstr_qent_spans=qstr_qent_spans)

        # These are mappings from Type predicates to the name of the type in the preprocessed data
        # Useful for (1) Finding starting rules that result in a particular type. (2) Finding the type of a logical prog.
        self.LANGTYPE_TO_TYPENAME_MAP = {
            PredicateType.get_type(Bool).name: hpcons.BOOL_TYPE
        }

        self.TYPENAME_TO_LANGTYPE_MAP = {
            hpcons.BOOL_TYPE: PredicateType.get_type(Bool).name
        }

        self.device_id = -1

        self._execution_parameters: ExecutorParameters = None

        self.bool_qstr_qent_func = None

        self.q_nemenspan2entidx = None

        # Shape: (QLen, Q_d)
        self.ques_encoded = None
        # Shape: (Qlen)
        self.ques_mask = None
        # Shape: (C, T, D)
        self.contexts = None
        # Shape: (C, D)
        self.contexts_vec = None
        # Shape: (C, T)
        self.contexts_mask = None
        # Shape: (E, C, M, 2)
        self.ne_ent_mens = None
        # Shape: (E, C, M, 2)
        self.num_ent_mens = None
        # Shape: (E, C, M, 2)
        self.date_ent_mens = None

        # Dict from QStr -> Idx into self.q_qstr_spans
        #         # self.q_qstr2idx = None
        #         # # Shape: (Num_of_Qstr, 2)
        #         # self.q_qstr_spans = None

        # Dict from Q_NE_men idx to EntityIdx corresonding to self.ne_ent_mens
        self.q_nemenspan2entidx = None

        # # Question encoded representation
        # self.ques_encoded = None
        # # Dictionary from QStr span to it's tensor representation
        # self.qstr2repr = None
        # Map from Qent span_str to LongTensor(2) of it's span in the question
        self.entidx2spans = None
        # Map from Qent span_str to LongTensor(Qlen) - a multi-hot vector with 1s at span token locations
        self.entidx2spanvecs = None

        self.entitymention_idxs_vec = None



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
        self.ques_encoded = kwargs["ques_encoded"]
        self.ques_mask = kwargs["ques_mask"]
        self.contexts = kwargs["contexts"]
        self.contexts_vec = kwargs["contexts_vec"]
        self.contexts_mask = kwargs["contexts_mask"]
        self.ne_ent_mens = kwargs["ne_ent_mens"]
        self.num_ent_mens = kwargs["num_ent_mens"]
        self.date_ent_mens = kwargs["date_ent_mens"]
        # self.q_qstr2idx= kwargs["q_qstr2idx"]
        # self.q_qstr_spans = kwargs["q_qstr_spans"]
        self.q_nemenspan2entidx = kwargs["q_nemenspan2entidx"]
        self.device_id = allenutil.get_device_of(self.ques_encoded)
        self.bool_qstr_qent_func = kwargs["bool_qstr_qent_func"]

        # Keep commented for use later
        # print(f"self.ques_embedded: {self.ques_embedded.size()}")
        # print(f"self.ques_mask: {self.ques_mask.size()}")
        # print(f"self.contexts: {self.contexts.size()}")
        # print(f"self.contexts_mask: {self.contexts_mask.size()}")
        # print(f"self.ne_ent_mens: {self.ne_ent_mens.size()}")
        # print(f"self.date_ent_mens: {self.date_ent_mens.size()}")
        # print(f"self.num_ent_mens: {self.num_ent_mens.size()}")
        # print(f"self.q_qstr2idx: {self.q_qstr2idx}")
        # print(f"self.q_qstr_spans: {self.q_qstr_spans.size()}")
        # print(f"self.q_nemenspan2entidx: {self.q_nemenspan2entidx}")


    def set_execution_parameters(self, execution_parameters: ExecutorParameters):
        self._execution_parameters: ExecutorParameters = execution_parameters

    def preprocess_arguments(self):
        """ Preprocessing arguments to make tensors that can be reused across logical forms during execution.

        For example, question span representations will be precomputed and stored here to reuse when executing different
        logical forms.
        """
        # self._preprocess_ques_representations()
        self._preprocess_ques_NE_menspans()


    def _make_gold_attentions(self):
        """ For questions of the kind: XXX E1 yyy E2 zzz zzz zzz

        We want
            Two qent attentions, one for E1 and E2
            One qstr attention, for zz zzz zzz

        Assume that the first entity is  Qent_1, second is Qent_2, and all tokens after that are Qstr
        """
        qlen = self.ques_mask.size()
        if self.device_id > -1:
            qent1 = torch.cuda.FloatTensor(qlen, device=self.device_id).fill_(0)
            qent2 = torch.cuda.FloatTensor(qlen, device=self.device_id).fill_(0)
            qstr = torch.cuda.FloatTensor(qlen, device=self.device_id).fill_(0)
        else:
            qent1 = torch.FloatTensor(qlen).fill_(0)
            qent2 = torch.FloatTensor(qlen).fill_(0)
            qstr = torch.FloatTensor(qlen).fill_(0)

        ne_mens = []
        for span in self.q_nemenspan2entidx.keys():
            # span is tokens@DELIM@START@END
            split_tokens = span.split(hpcons.SPAN_DELIM)
            start, end = int(split_tokens[-2]), int(split_tokens[-1])
            ne_mens.append((start, end))
        # Sort based on start token
        sorted_ne_mens = sorted(ne_mens, key=lambda x: x[0])

        # Assume that first mention is E1 and second is E2.
        if len(sorted_ne_mens) < 2:
            e1 = sorted_ne_mens[0]
            e2 = sorted_ne_mens[0]
        else:
            e1 = sorted_ne_mens[0]
            e2 = sorted_ne_mens[1]

        qent1[e1[0]:e1[1] + 1] = 1.0
        qent2[e2[0]:e2[1] + 1] = 1.0
        qstr[e2[1] + 1:] = 1.0

        qent1 = qent1 * self.ques_mask
        qent2 = qent2 * self.ques_mask
        qstr = qstr * self.ques_mask

        return qent1, qent2, qstr


    ''' This function is deprecated now that we pass in question_reprs and don't have ques_spans to represent
    def _preprocess_ques_representations(self):
        # Embedding the complete question
        ques_encoded_ex = self._execution_parameters._ques_encoder(self.ques_encoded.unsqueeze(0),
                                                                   self.ques_mask.unsqueeze(0))
        # Shape: (Qlen, Q_d)
        self.ques_encoded = ques_encoded_ex.squeeze(0)

        # For all QSTR, computing repr by running ques_encoder, and concatenating repr of end-points.
        self.qstr2repr = {}
        for qstr, qstr_idx in self.q_qstr2idx.items():
            # Shape: [2] denoting the span in the question
            qstr_span = self.q_qstr_spans[qstr_idx]
            # Shape: (QSTR_len, emb_dim)
            qstr_embedded = self.ques_encoded[qstr_span[0]:qstr_span[1] + 1]
            qstr_mask = self.ques_mask[qstr_span[0]:qstr_span[1] + 1]
            # [1, QSTR_len, emb_dim]
            qstr_encoded_ex = self._execution_parameters._ques_encoder(qstr_embedded.unsqueeze(0),
                                                                       qstr_mask.unsqueeze(0))
            # Shape: [QSTR_len, emb_dim]
            qstr_encoded = qstr_encoded_ex.squeeze(0)
            # Concatenating first and last time step of encoded qstr
            # Shape: (2 * Qd)
            qstr_repr = torch.cat([qstr_encoded[0].unsqueeze(0), qstr_encoded[-1].unsqueeze(0)], 1).squeeze(0)
            self.qstr2repr[qstr] = qstr_repr

    '''

    def _preprocess_ques_NE_menspans(self):
        """ Preprocess Ques NE mens to extract spans for each of the entity mentioned.
        Makes two dictionaries, (1) Stores a list of mention spans (Tensor(2)) in the question for each entity
        mentioned. (2) For each entity mentioned, a list of binary-vectors the size of Qlen, with 1s for tokens in the
        span. For example, if the question (len=5) has two mentions, [1,2] and [3,4] linking to entities e1 and e2.
        The two dictionaries made will be {e1: [Tensor([1,2])], e2: [Tensor([3,4])]} and
        {e1: [FloatTensor([0,..,1,1,.0])], e2: [FloatTensor([0...,0,1,1])]}

        Additionally, self.entitymention_idxs_vec -- a vector of QLen is made that contains 1 at all locations
        that are NE mens is made; this is used for Qent attention loss.
        """
        q_len = self.ques_mask.size()
        # Use the qent attention to get a distribution over entities.
        # Use the self.q_nemenspan2entidx map to extract spans for entities in the question.
        # Since an entity can be mentioned multiple times, we'll create a dictionary from entityidx2spans
        self.entidx2spans = {}
        self.entidx2spanvecs = {}

        if self.device_id > -1:
            self.entitymention_idxs_vec = torch.cuda.FloatTensor(q_len, device=self.device_id).fill_(0)
        else:
            self.entitymention_idxs_vec = torch.FloatTensor(q_len).fill_(0)

        for span, entity_idx in self.q_nemenspan2entidx.items():
            # Span: QENT:TOKEN_1@DELIM@TOKEN_2...TOKEN_N@DELIM@SPAN_START@DELIM@SPAN_END
            # Start and end are inclusive
            split_tokens = span.split(hpcons.SPAN_DELIM)
            start, end = int(split_tokens[-2]), int(split_tokens[-1])
            if self.device_id > -1:
                span_tensor = torch.cuda.LongTensor([start, end], device=self.device_id)
                onehot_span_tensor = torch.cuda.FloatTensor(q_len, device=self.device_id).fill_(0)
                onehot_span_tensor[start:end + 1] = 1
            else:
                span_tensor = torch.LongTensor([start, end])
                onehot_span_tensor = torch.FloatTensor(q_len).fill_(0)
                onehot_span_tensor[start:end + 1] = 1
            if entity_idx not in self.entidx2spans:
                self.entidx2spans[entity_idx] = []
                self.entidx2spanvecs[entity_idx] = []
            self.entidx2spans[entity_idx].append(span_tensor)
            self.entidx2spanvecs[entity_idx].append(onehot_span_tensor)
            self.entitymention_idxs_vec += onehot_span_tensor


    @predicate_with_side_args(['question_attention'])
    def find_Qstr(self, question_attention: torch.FloatTensor) -> Qstr:
        return Qstr(attention=question_attention)


    @predicate_with_side_args(['question_attention'])
    def find_Qent(self, question_attention: torch.FloatTensor) -> Qent:
        return Qent(attention=question_attention)


    @predicate
    def bool_qent_qstr(self, qent: Qent, qstr: Qstr) -> Bool1:
        returnval = None
        if self.bool_qstr_qent_func == 'mentions':
            returnval = self.bool_qent_qstr_wmens(qent=qent, qstr=qstr)
        elif self.bool_qstr_qent_func == 'context':
            returnval = self.bool_qent_qstr_wcontext(qent=qent, qstr=qstr)
        else:
            raise NotImplementedError

        return returnval


    def bool_qent_qstr_wcontext(self, qent: Qent, qstr: Qstr) -> Bool1:
        qent_att = qent._value * self.ques_mask
        qstr_att = qstr._value * self.ques_mask

        # Shape: (Q_d)
        qstr_repr = (self.ques_encoded * qstr_att.unsqueeze(1)).sum(0)

        # Computing score/distribution over all entities mentioned in the question based on the qent ques_attention
        # Entity_prob is proportional to the sum of the attention values of the tokens in the entity's mention
        entidx2prob = {}
        entidx2score = {}
        total_score = 0.0
        for entidx, span_onehot_vecs in self.entidx2spanvecs.items():
            entity_score = sum([(spanvec * qent_att).sum() for spanvec in span_onehot_vecs])
            total_score += entity_score
            entidx2score[entidx] = entity_score
            # entidx2prob[entidx] = entity_score
        for entidx, entity_score in entidx2score.items():
            entidx2prob[entidx] = entity_score / total_score

        # Find the answer prob based on each entity mentioned in the question, and compute the expection later ...
        entidx2ansprob = {}
        for entidx, span_onehot_vecs in self.entidx2spanvecs.items():
            # Shape: (Qlen)
            entity_token_att = sum([(spanvec * qent_att) for spanvec in span_onehot_vecs])
            # Shape: Q_d
            qent_repr = (self.ques_encoded * entity_token_att.unsqueeze(1)).sum(0)
            # Shape: 2*Q_d
            qent_qstr_repr = torch.cat([qent_repr, qstr_repr])

            # Now dot prod with context vecs
            # Shape: (C, 2 * D)
            context_vecs = torch.cat([self.contexts_vec, self.contexts_vec], 1)

            # Shape: C
            dot_prod = self._execution_parameters._bool_bilinear(qent_qstr_repr, self.contexts_vec)
            # dot_prod = (qent_qstr_repr.unsqueeze(0) * context_vecs).sum(1)

            probs = torch.sigmoid(dot_prod)

            boolean_prob = torch.max(probs)  # .unsqueeze(0)
            entidx2ansprob[entidx] = boolean_prob

        # Computing the expected boolean_answer based on the qent probs
        expected_prob = 0.0
        for entidx, ent_ans_prob in entidx2ansprob.items():
            # ent_prob = entidx2prob[entidx]
            ent_prob = entidx2score[entidx]
            expected_prob += ent_prob * ent_ans_prob

        return Bool1(value=expected_prob)

    def bool_qent_qstr_wmens(self, qent: Qent, qstr: Qstr) -> Bool1:
        qent_att = qent._value * self.ques_mask
        qstr_att = qstr._value * self.ques_mask

        # Shape: (Q_d)
        qstr_repr = (self.ques_encoded * qstr_att.unsqueeze(1)).sum(0)

        # Computing score/distribution over all entities mentioned in the question based on the qent ques_attention
        # Entity_prob is proportional to the sum of the attention values of the tokens in the entity's mention
        entidx2prob = {}
        entidx2score = {}
        total_score = 0.0
        for entidx, span_onehot_vecs in self.entidx2spanvecs.items():
            entity_score = sum([(spanvec * qent_att).sum() for spanvec in span_onehot_vecs])
            total_score += entity_score
            entidx2score[entidx] = entity_score
            # entidx2prob[entidx] = entity_score
        for entidx, entity_score in entidx2score.items():
            entidx2prob[entidx] = entity_score/total_score

        # Find the answer prob based on each entity mentioned in the question, and compute the expection later ...
        entidx2ansprob = {}
        for entidx in entidx2prob.keys():
            # Shape: (C, M, 2)
            qent_mens = self.ne_ent_mens[entidx]
            qent_mens_mask = (qent_mens[..., 0] >= 0).long()

            # Shape: (C, M, 2*D)
            qent_men_repr = self._execution_parameters._span_extractor(self.contexts, qent_mens,
                                                                       self.contexts_mask,
                                                                       qent_mens_mask)
            q_repr_cat = torch.cat([qstr_repr, qstr_repr], 0)
            qstr_repr_ex = q_repr_cat.unsqueeze(0).unsqueeze(1)
            scores = torch.sum(qstr_repr_ex * qent_men_repr, 2)
            probs = torch.sigmoid(scores) * qent_mens_mask.float()

            boolean_prob = torch.max(probs)  # .unsqueeze(0)
            entidx2ansprob[entidx] = boolean_prob

        # Computing the expected boolean_answer based on the qent probs
        expected_prob = 0.0
        for entidx, ent_ans_prob in entidx2ansprob.items():
            # ent_prob = entidx2prob[entidx]
            ent_prob = entidx2score[entidx]
            expected_prob += ent_prob * ent_ans_prob

        return Bool1(value=expected_prob)


    @predicate
    def bool_and(self, bool1: Bool1, bool2: Bool1) -> Bool:
        """ AND operation between two booleans """
        # return_val = bool1._value * bool2._value
        return_val = torch.min(torch.cat([bool1._value.unsqueeze(0), bool2._value.unsqueeze(0)], 0))
        # return_val = torch.sigmoid(10.0*bool1._value + 10.0*bool2._value - 15.0)
        return Bool(value=return_val)
        # return Bool(value=bool1._value * bool2._value)

    # @predicate
    # def bool_or(self, bool1: Bool1, bool2: Bool1) -> Bool:
    #     """ OR operation between two booleans """
    #     return_val = torch.max(torch.cat([bool1._value.unsqueeze(0), bool2._value.unsqueeze(0)], 0))
    #     return Bool(value=return_val)


if __name__=="__main__":

    # b = Bool(value=torch.FloatTensor([0.65]))
    # print(b._value)
    # print(b._bool_val)
    # print(PredicateType.get_type(b.__class__).name)
    # print(PredicateType.get_type(Bool).name)

    language = HotpotQALanguage(qstr_qent_spans=["QSTR:Hi", "QENT:Nitish"])
    print(language._functions)

    all_prods = language.all_possible_productions()

    print("All prods:\n{}\n".format(all_prods))
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




