import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import math
import copy

from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
import allennlp.nn.util as allenutil
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models.reading_comprehension.util import get_best_span

from allennlp.training.metrics import Average, DropEmAndF1

from semqa.domain_languages.drop_language import Date, QuestionSpanAnswer, PassageSpanAnswer

import utils.util as myutils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# In this three tuple, add "QENT:qent QSTR:qstr" in the twp gaps, and join with space to get the logical form
GOLD_BOOL_LF = ("(bool_and (bool_qent_qstr ", ") (bool_qent_qstr", "))")


def getGoldLF(qent1_action, qent2_action, qstr_action):
    qent1, qent2, qstr = (qent1_action.split(" -> ")[1], qent2_action.split(" -> ")[1], qstr_action.split(" -> ")[1])
    # These qent1, qent2, and qstr are actions
    return f"{GOLD_BOOL_LF[0]} {qent1} {qstr}{GOLD_BOOL_LF[1]} {qent2} {qstr}{GOLD_BOOL_LF[2]}"


@Model.register("drop_qanet")
class QANet(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        num_highway_layers: int,
        phrase_layer: Seq2SeqEncoder,
        matrix_attention_layer: MatrixAttention,
        modeling_layer: Seq2SeqEncoder,
        dropout: float = 0.0,
        debug: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super(QANet, self).__init__(vocab=vocab, regularizer=regularizer)

        question_encoding_dim = phrase_layer.get_output_dim()
        self._text_field_embedder = text_field_embedder

        text_embed_dim = self._text_field_embedder.get_output_dim()

        encoding_in_dim = phrase_layer.get_input_dim()
        encoding_out_dim = phrase_layer.get_output_dim()
        modeling_in_dim = modeling_layer.get_input_dim()
        modeling_out_dim = modeling_layer.get_output_dim()

        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)
        self._phrase_layer = phrase_layer

        self._matrix_attention = matrix_attention_layer

        self._modeling_proj_layer = torch.nn.Linear(encoding_out_dim * 4, modeling_in_dim)
        self._modeling_layer = modeling_layer

        self._span_start_predictor = torch.nn.Linear(modeling_out_dim * 2, 1)
        self._span_end_predictor = torch.nn.Linear(modeling_out_dim * 2, 1)

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self.modelloss_metric = Average()
        self._drop_metrics = DropEmAndF1()

        initializer(self)

    @overrides
    def forward(
        self,
        question: Dict[str, torch.LongTensor],
        passage: Dict[str, torch.LongTensor],
        passageidx2numberidx: torch.LongTensor,
        passage_number_values: List[int],
        passageidx2dateidx: torch.LongTensor,
        passage_date_values: List[List[Date]],
        actions: List[List[ProductionRule]],
        datecomp_ques_event_date_groundings: List[Tuple[List[int], List[int]]] = None,
        numcomp_qspan_num_groundings: List[Tuple[List[int], List[int]]] = None,
        strongly_supervised: List[bool] = None,
        qtypes: List[str] = None,
        qattn_supervision: torch.FloatTensor = None,
        answer_types: List[str] = None,
        answer_as_passage_spans: torch.LongTensor = None,
        answer_as_question_spans: torch.LongTensor = None,
        epoch_num: List[int] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

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

        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = allenutil.masked_softmax(
            passage_question_similarity, question_mask, memory_efficient=True
        )
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = allenutil.weighted_sum(encoded_question, passage_question_attention)

        # Shape: (batch_size, question_length, passage_length)
        question_passage_attention = allenutil.masked_softmax(
            passage_question_similarity.transpose(1, 2), passage_mask, memory_efficient=True
        )
        # Shape: (batch_size, passage_length, passage_length)
        attention_over_attention = torch.bmm(passage_question_attention, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_passage_vectors = allenutil.weighted_sum(encoded_passage, attention_over_attention)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        merged_passage_attention_vectors = self._dropout(
            torch.cat(
                [
                    encoded_passage,
                    passage_question_vectors,
                    encoded_passage * passage_question_vectors,
                    encoded_passage * passage_passage_vectors,
                ],
                dim=-1,
            )
        )

        modeled_passage_list = [self._modeling_proj_layer(merged_passage_attention_vectors)]

        for _ in range(3):
            modeled_passage = self._dropout(self._modeling_layer(modeled_passage_list[-1], passage_mask))
            modeled_passage_list.append(modeled_passage)

        # Shape: (batch_size, passage_length, modeling_dim * 2))
        span_start_input = torch.cat([modeled_passage_list[-3], modeled_passage_list[-2]], dim=-1)
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)

        # Shape: (batch_size, passage_length, modeling_dim * 2)
        span_end_input = torch.cat([modeled_passage_list[-3], modeled_passage_list[-1]], dim=-1)
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        span_start_logits = allenutil.replace_masked_values(span_start_logits, passage_mask, -1e32)
        span_end_logits = allenutil.replace_masked_values(span_end_logits, passage_mask, -1e32)

        # Shape: (batch_size, passage_length)
        span_start_probs = torch.nn.functional.softmax(span_start_logits, dim=-1)
        span_end_probs = torch.nn.functional.softmax(span_end_logits, dim=-1)

        span_start_logprob = allenutil.masked_log_softmax(span_start_logits, mask=passage_mask, dim=-1)
        span_end_logprob = allenutil.masked_log_softmax(span_end_logits, mask=passage_mask, dim=-1)
        span_start_logprob = allenutil.replace_masked_values(span_start_logprob, passage_mask, -1e32)
        span_end_logprob = allenutil.replace_masked_values(span_end_logprob, passage_mask, -1e32)

        best_span = get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "passage_question_attention": passage_question_attention,
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
        }

        if answer_types is not None:
            loss = 0
            for i in range(batch_size):
                loss += self._get_span_answer_log_prob(
                    answer_as_spans=answer_as_passage_spans[i],
                    span_log_probs=(span_start_logprob[i], span_end_logprob[i]),
                )

            loss = (-1.0 * loss) / batch_size

            self.modelloss_metric(myutils.tocpuNPList(loss)[0])
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["best_span_str"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]["question_tokens"])
                passage_tokens.append(metadata[i]["passage_tokens"])
                passage_str = metadata[i]["original_passage"]
                offsets = metadata[i]["passage_token_offsets"]
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict["best_span_str"].append(best_span_string)
                answer_annotations = metadata[i].get("answer_annotation")
                self._drop_metrics(best_span_string, [answer_annotations])

        output_dict.update({"metadata": metadata})

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}
        model_loss = self.modelloss_metric.get_metric(reset)
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        metric_dict.update({"em": exact_match, "f1": f1_score, "model_loss": model_loss})

        return metric_dict

    @staticmethod
    def _get_span_answer_log_prob(
        answer_as_spans: torch.LongTensor, span_log_probs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
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
        gold_passage_span_starts = answer_as_spans[:, :, 0]
        gold_passage_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_passage_span_mask = (gold_passage_span_starts != -1).long()
        clamped_gold_passage_span_starts = allenutil.replace_masked_values(
            gold_passage_span_starts, gold_passage_span_mask, 0
        )
        clamped_gold_passage_span_ends = allenutil.replace_masked_values(
            gold_passage_span_ends, gold_passage_span_mask, 0
        )
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = torch.gather(span_start_log_probs, 1, clamped_gold_passage_span_starts)
        log_likelihood_for_span_ends = torch.gather(span_end_log_probs, 1, clamped_gold_passage_span_ends)

        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = allenutil.replace_masked_values(
            log_likelihood_for_spans, gold_passage_span_mask, -1e7
        )

        # Shape: (batch_size, )
        log_marginal_likelihood_for_span = allenutil.logsumexp(log_likelihood_for_spans)

        return log_marginal_likelihood_for_span

    @staticmethod
    def find_valid_start_states(
        answer_as_question_spans: torch.LongTensor, answer_as_passage_spans: torch.LongTensor, batch_size: int
    ) -> List[Set[Any]]:
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
    def _get_best_spans(
        batch_denotations,
        batch_denotation_types,
        question_char_offsets,
        question_strs,
        passage_char_offsets,
        passage_strs,
        *args,
    ):
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
                best_span = get_best_span(
                    span_end_logits=denotation._value[0].unsqueeze(0),
                    span_start_logits=denotation._value[1].unsqueeze(0),
                ).squeeze(0)
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

    def datecompare_goldattn_to_sideargs(
        self,
        batch_actionseqs: List[List[List[str]]],
        batch_actionseq_sideargs: List[List[List[Dict]]],
        batch_gold_attentions: List[Tuple[torch.Tensor, torch.Tensor]],
    ):

        for ins_idx in range(len(batch_actionseqs)):
            instance_programs = batch_actionseqs[ins_idx]
            instance_prog_sideargs = batch_actionseq_sideargs[ins_idx]
            instance_gold_attentions = batch_gold_attentions[ins_idx]
            for program, side_args in zip(instance_programs, instance_prog_sideargs):
                first_qattn = True  # This tells model which qent attention to use
                # print(side_args)
                # print()
                for action, sidearg_dict in zip(program, side_args):
                    if action == "PassageAttention -> find_PassageAttention":
                        if first_qattn:
                            sidearg_dict["question_attention"] = instance_gold_attentions[0]
                            first_qattn = False
                        else:
                            sidearg_dict["question_attention"] = instance_gold_attentions[1]

    def get_date_compare_ques_attns(self, qstr: str, masked_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Question only has one 'or'
            Attn2 is after or until ?
            Attn1 is after first ',' or ':' ('first', 'last', 'later') until the 'or'
        """

        or_split = qstr.split(" or ")
        assert len(or_split) == 2

        tokens = qstr.split(" ")

        attn_1 = torch.cuda.FloatTensor(masked_len, device=0).fill_(0.0)
        attn_2 = torch.cuda.FloatTensor(masked_len, device=0).fill_(0.0)

        or_idx = tokens.index("or")
        # Last token is ? which we don't want to attend to
        attn_2[or_idx + 1 : len(tokens) - 1] = 1.0
        attn_2 = attn_2 / attn_2.sum()

        # Gets first index of the item
        try:
            comma_idx = tokens.index(",")
        except:
            comma_idx = 100000
        try:
            colon_idx = tokens.index(":")
        except:
            colon_idx = 100000

        try:
            hyphen_idx = tokens.index("-")
        except:
            hyphen_idx = 100000

        split_idx = min(comma_idx, colon_idx, hyphen_idx)

        if split_idx == 100000 or (or_idx - split_idx <= 1):
            # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
            if "first" in tokens:
                split_idx = tokens.index("first")
            elif "second" in tokens:
                split_idx = tokens.index("second")
            elif "last" in tokens:
                split_idx = tokens.index("last")
            elif "later" in tokens:
                split_idx = tokens.index("later")
            else:
                split_idx = -1

        assert split_idx != -1, f"{qstr} {split_idx} {or_idx}"

        attn_1[split_idx + 1 : or_idx] = 1.0
        attn_1 = attn_1 / attn_1.sum()

        return attn_1, attn_2

    def get_gold_quesattn_datecompare(self, metadata, masked_len) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = len(metadata)

        gold_ques_attns = []

        for i in range(batch_size):
            question_tokens: List[str] = metadata[i]["question_tokens"]
            qstr = " ".join(question_tokens)
            assert len(qstr.split(" ")) == len(question_tokens)

            gold_ques_attns.append(self.get_date_compare_ques_attns(qstr, masked_len))

        return gold_ques_attns
