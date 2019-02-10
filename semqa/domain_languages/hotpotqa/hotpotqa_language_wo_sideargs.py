from typing import Callable, List, Dict, Tuple
import logging
from functools import cmp_to_key

import torch
import torch.nn.functional
from torch import Tensor

from allennlp.common.registrable import Registrable
import allennlp.nn.util as allenutil
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
from allennlp.modules.similarity_functions import SimilarityFunction, BilinearSimilarity
from allennlp.semparse import (DomainLanguage, ExecutionError, ParsingError,
                               predicate, predicate_with_side_args, util)
from allennlp.semparse.domain_languages.domain_language import PredicateType

from semqa.domain_languages.hotpotqa.execution_params import ExecutorParameters
from semqa.domain_languages.hotpotqa.hotpotqa_language import Qstr, Qent, Bool, Bool1, HotpotQALanguage

import datasets.hotpotqa.utils.constants as hpcons


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class HotpotQALanguageWOSideArgs(HotpotQALanguage):
    def __init__(self, qstr_qent_spans: List[str]):
        super().__init__(start_types={Bool})

        self._add_constants(qstr_qent_spans=qstr_qent_spans)

    def _add_constants(self, qstr_qent_spans: List[str]):
        for span in qstr_qent_spans:
            if span.startswith(hpcons.QSTR_PREFIX):
                self.add_constant(name=span, value=Qstr(span), type_=Qstr)
            elif span.startswith(hpcons.QENT_PREFIX):
                # Question NE men span
                self.add_constant(name=span, value=Qent(span), type_=Qent)


    def preprocess_arguments(self):
        """ Preprocessing arguments to make tensors that can be reused across logical forms during execution.

        For example, question span representations will be precomputed and stored here to reuse when executing different
        logical forms.
        """
        self._preprocess_ques_representations()

    def _preprocess_ques_representations(self):
        # Embedding the complete question
        # For all QSTR, computing repr by running ques_encoder, and concatenating repr of end-points
        self.qstr2repr = {}
        for qstr, qstr_idx in self.quesspans2idx.items():
            # Shape: [2] denoting the span in the question
            qstr_span = self.quesspans_spans[qstr_idx]
            # Shape: (QSTR_len, emb_dim)
            qstart = self.ques_encoded[qstr_span[0]]
            qend = self.ques_encoded[qstr_span[1]]

            # Shape: (2 * Qd)
            qstr_repr = torch.cat([qstart, qend], 0).squeeze(0)
            self.qstr2repr[qstr] = qstr_repr


    def _get_gold_actions(self) -> Tuple[str, str, str]:
        def getspanfromaction(actionstr):
            """ Action string of the type Qent -> QENT:Tokens@DELIM@START@DELIM@END
                Return (START, END)
            """
            splittokens = actionstr.split(hpcons.SPAN_DELIM)
            start, end = int(splittokens[-2]), int(splittokens[-1])
            return (start, end)

        def tuplecomparision(t1, t2):
            # when arranging in ascending, longest with greater end should come first
            len1 = t1[1] - t1[0]
            len2 = t2[1] - t2[0]
            if len2 < len1:
                return -1
            elif len1 < len2:
                return 1
            else:
                if t1[1] < t2[1]:
                    return 1
                elif t1[1] > t2[1]:
                    return -1
                else:
                    return 0
        # def tuplecomparision(t1, t2):
        #     # When arranging in ascending, lower start but longest within should come first
        #     if t1[0] < t2[0]:
        #         return -1
        #     elif t1[0] > t2[0]:
        #         return 1
        #     else:
        #         if t1[1] < t2[1]:
        #             return 1
        #         elif t1[1] > t2[1]:
        #             return -1
        #         else:
        #             return 0



        """ For boolean questions of the kind: XXX E1 yyy E2 zzz zzz zzz

        We want
            Two Qent -> QENT:Tokens@DELIM@START@DELIM@END' style actions
            One Qstr -> QSTR:Tokens@DELIM@START@DELIM@END' action for zz zzz zzz

        Assume that the first two entities are first/second Qent actions, and longest span that starts right after
        second entity is the Qstr action
        """
        qent_actions = self.get_nonterminal_productions()['Qent']
        qstr_actions = self.get_nonterminal_productions()['Qstr']

        # Dict from span tuple to action str. this will make sorting easier.
        span2qentactions = {}
        span2qstractions = {}
        for a in qent_actions:
            span = getspanfromaction(a)
            span2qentactions[span] = a
        for a in qstr_actions:
            span = getspanfromaction(a)
            span2qstractions[span] = a

        sorted_qentspans = sorted(span2qentactions.keys(),
                                  key=cmp_to_key(tuplecomparision),
                                  reverse=False)
        sorted_qstrspans = sorted(span2qstractions.keys(),
                                  key=cmp_to_key(tuplecomparision),
                                  reverse=False)
        if len(sorted_qentspans) >= 2:
            gold_qent1span = sorted_qentspans[0]
            gold_qent2span = sorted_qentspans[1]
        else:
            gold_qent1span = sorted_qentspans[0]
            gold_qent2span = sorted_qentspans[0]


        gold_qent2end = gold_qent2span[1]
        gold_qstr_span = None
        for qstrspan in sorted_qstrspans:
            if qstrspan[0] > gold_qent2end:
                gold_qstr_span = qstrspan
                break

        if gold_qstr_span is None:
            gold_qstr_span = sorted_qstrspans[-1]

        qent1_action = span2qentactions[gold_qent1span]
        qent2_action = span2qentactions[gold_qent2span]
        qstr_action = span2qstractions[gold_qstr_span]

        return qent1_action, qent2_action, qstr_action


    @predicate
    def bool_qent_qstr(self, qent: Qent, qstr: Qstr) -> Bool1:
        returnval = None
        if self.bool_qstr_qent_func == 'mentions':
            # returnval = self.bool_qent_qstr_wmens(qent=qent, qstr=qstr)
            raise NotImplementedError
        elif self.bool_qstr_qent_func == 'context':
            returnval = self.bool_qent_qstr_wcontext(qent_obj=qent, qstr_obj=qstr)
        else:
            raise NotImplementedError

        return returnval


    def bool_qent_qstr_wcontext(self, qent_obj: Qent, qstr_obj: Qstr) -> Bool1:
        # qent_att = qent._value * self.ques_mask
        # qstr_att = qstr._value * self.ques_mask

        qent = qent_obj._value
        qstr = qstr_obj._value

        # Shape: 2 * D
        qstr_repr = self.qstr2repr[qstr]
        qent_repr = self.qstr2repr[qent]

        # Shape: 4 * Q_d
        qent_qstr_repr = torch.cat([qent_repr, qstr_repr])
        # Shape: C
        ques_context_sim = self._execution_parameters._bool_bilinear(qent_qstr_repr, self.contexts_vec)

        probs = torch.sigmoid(ques_context_sim)
        boolean_prob = torch.max(probs)
        return Bool1(value=boolean_prob)

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


if __name__=="__main__":

    # b = Bool(value=torch.FloatTensor([0.65]))
    # print(b._value)
    # print(b._bool_val)
    # print(PredicateType.get_type(b.__class__).name)
    # print(PredicateType.get_type(Bool).name)

    language = HotpotQALanguageWOSideArgs(qstr_qent_spans=["QSTR:Hi", "QENT:Nitish"])
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




