from typing import Dict, List, Any, TypeVar, Tuple
import json
import logging
import copy

import numpy as np

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField
from allennlp.data.fields import ProductionRuleField, MetadataField, SpanField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from semqa.worlds.hotpotqa.sample_world import SampleHotpotWorld
from semqa.domain_languages.hotpotqa.hotpotqa_language import HotpotQALanguage
from semqa.data.datatypes import DateField, NumberField
import datasets.hotpotqa.utils.constants as hpconstants
import utils.util as util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sample_hotpot")
class SampleHotpotDatasetReader(DatasetReader):
    """
    Copied from NlvrDatasetReader
    (https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/semantic_parsing/nlvr.py)
    """

    """
    Instance-specific-actions (linked actions):
    We currently deal with linked actions that lead to terminals. For example "QSTR -> ques_span"
    Much goes behind making such a rule possible, and some decisions need to be taken.
    
    1. The right-hand-side gets a name, this is used in the World to add to the name_mapping and type_signatures.
    We will add the prefix of the acrtion_type to the actual context.
    For example QSTR:ques_span_tokens, QENT:ques_mention_tokens
    
    2. After getting added to the world, in the reader it is added as a non-global production rule
    
    3. linked_rule_idx: Current reader implementation make a dictionary from RHS: idx to store other data
    about rule in lists. See below.
    
    4. Linking_score: For each action, a binary vector of length as ques_length is made to score the action.
    Passed as a list, the idx is from linked_rule_idx (3.)
    
    5. Output_embedding: An action needs an output_embedding if selected in the decoder to feed to the next time step. 
    Since all our instance-specific-actions are spans, we use span_embeddings for which we pass a list of SpanField
    Idxs from linked_rule_idx (3.)
    
    6. Denotation: This is the representation used in the execution.
    For example, QSTR might need a span embedding for which 5. can be used.
    QENT will need a vector the size of the number of NE entities in the instance.
    For QSTR: Pass Dict from {ques_span: idx} indexing into the SpanField list of 5.
    For QENT: Pass Dict from {ques_span: idx} indexing into a ArrayField wiht shape: (Num_QENT_spans, Num_NE_entities)

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    tokenizer : ``Tokenizer`` (optional)
        The tokenizer used for sentences in NLVR. Default is ``WordTokenizer``
    sentence_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for tokens in input sentences.
        Default is ``{"tokens": SingleIdTokenIndexer()}``
    nonterminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for non-terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    terminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    output_agendas : ``bool`` (optional)
        If preparing data for a trainer that uses agendas, set this flag and the datset reader will
        output agendas.
    """
    def __init__(self,
                 lazy: bool = False,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._sentence_token_indexers = sentence_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._nonterminal_indexers = nonterminal_indexers or {"tokens":
                                                              SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = terminal_indexers or {"tokens": SingleIdTokenIndexer("rule_labels")}

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            logger.info(f"Reading instances from lines in file: {file_path}")
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                data = json.loads(line)

                # space delimited tokenized
                question = data[hpconstants.q_field]
                # List of question mentions. Stored as mention-tuples --- (text, start, end, label)
                # TODO(nitish): Fix this to include all types
                q_nemens = data[hpconstants.q_ent_ner_field]
                # List of (title, space_delimited_tokenized_contexts)
                contexts = data[hpconstants.context_field]
                # List of list --- For each context , list of mention-tuples as (text, start, end, label)
                contexts_ent_ners = data[hpconstants.context_ent_ner_field]
                contexts_num_ners = data[hpconstants.context_num_ner_field]
                contexts_date_ners = data[hpconstants.context_date_ner_field]

                # Mention to entity mapping -- used to make the grounding vector
                context_entmens2entidx = data[hpconstants.context_entmens2entidx]
                context_nummens2entidx = data[hpconstants.context_nummens2entidx]
                context_datemens2entidx = data[hpconstants.context_datemens2entidx]

                # Entity to mentions --- Used to find the number of entities of each type in the contexts
                context_eqent2entmens = data[hpconstants.context_eqent2entmens]
                context_eqent2nummens = data[hpconstants.context_eqent2nummens]
                context_eqent2datemens = data[hpconstants.context_eqent2datemens]

                # Dict from {date_string: (date, month, year)} normalization. -1 indicates invalid field
                dates_normalized_dict = data[hpconstants.dates_normalized_field]
                # Dict from {num_string: float_val} normalization.
                nums_normalized_dict = data[hpconstants.nums_normalized_field]
                # Dict from {ent_idx: [(context_idx, men_idx)]} --- output pf CDCR

                # Grounding of ques entity mentions
                qnemens_to_ent = data[hpconstants.q_entmens2entidx]

                ans_type = None
                ans_grounding = None
                if hpconstants.ans_type_field in data:
                    ans_type = data[hpconstants.ans_type_field]
                    ans_grounding = data[hpconstants.ans_grounding_field]

                instance = self.text_to_instance(question,
                                                 q_nemens,
                                                 contexts,
                                                 contexts_ent_ners,
                                                 contexts_num_ners,
                                                 contexts_date_ners,
                                                 context_entmens2entidx,
                                                 context_nummens2entidx,
                                                 context_datemens2entidx,
                                                 context_eqent2entmens,
                                                 context_eqent2nummens,
                                                 context_eqent2datemens,
                                                 dates_normalized_dict,
                                                 nums_normalized_dict,
                                                 qnemens_to_ent,
                                                 ans_type,
                                                 ans_grounding)
                if instance is not None:
                    yield instance


    def make_span_str(self, span_tokens: List[str], start_idx: int, end_idx: int, prefix: str) -> str:
        """ Make span str that will be used for in the rule names (QSTR -> Span_Str), and also
            in dictionaries mapping to linking_score, span_fields, etc.

        The span_str is a "prefix_spantoken1_delimeter_spantoken2_delimeter_..._spanstart_delimiter_spanend"

        end_idx is exclusive
        """

        delim = hpconstants.SPAN_DELIM
        span_str = prefix + delim.join(span_tokens) + delim + str(start_idx) + delim + str(end_idx)
        return span_str


    def spanIntersectWithAnyOneSpanInList(self, span, list_of_spans):
        """ Returns true if given span overlaps with any span in a given list.

        All spans have exclusive ends
        """

        intersects = False
        for test_span in list_of_spans:
            if util.isSpanOverlap(span, test_span):
                intersects = True

        return intersects



    def get_ques_spans(self, ques_tokens: List[str], q_ent_ners: List[Tuple], ques_textfield: TextField):
        """ Make question spans (delimited by constants.SPAN_DELIM) and also return their spans (inclusive)
        Current implementation: Take unigrams, bigrams, and trigrams that don't overlap with NE mens

        Parameters:
        -----------
        ques_tokens: List[str]: List of question tokens
        q_ent_ners: List[Tuple] List of q NE ner mens
        ques_textfield: TextField: For SpanFields

        Returns:
        --------
        ques_spans: `List[str]`
            All question spans delimited by constants.SPAN_DELIM
        ques_span2idx: ``Dict[str]: int``
            Dict mapping from q_span to Idx
        ques_spans_linking_score: ``List[List[float]]``
            For each q_span, Binary list the size of q_length indicating q_token's presence in the span
            This is used to score the q_span action in the decoder based on attention
        ques_spans_spanfields: List[SpanField]
            List of SpanField for each q_span. One use case is to get an output embedding for the decoder.
            Can possibly be used as input embedding as well (will need to figure out the TransitionFunction)
        """

        # This list will contain all possible spans (unigrams, bigrams ...) as (start,end) tuples
        # start and end will inclusive
        ques_spans_idxs = []

        qlen = len(ques_tokens)
        unigram_spans = [(i, i+1) for i in range(0, qlen)]
        bigram_spans = [(i, i+2) for i in range(0, qlen-1)]
        trigam_spans = [(i, i+3) for i in range(0, qlen-2)]

        ne_mens_spans = [(men[1], men[2]) for men in q_ent_ners]

        # Only keep spans that don't intersect with ques NE mens
        # Only keep bigrams and trigrams (if possible)
        for span in bigram_spans:
            if not self.spanIntersectWithAnyOneSpanInList(span, ne_mens_spans):
                ques_spans_idxs.append(span)

        for span in trigam_spans:
            if not self.spanIntersectWithAnyOneSpanInList(span, ne_mens_spans):
                ques_spans_idxs.append(span)

        # Add unigram spans only if no bigrams and trigrams present
        if len(ques_spans_idxs) == 0:
            for span in unigram_spans:
                if not self.spanIntersectWithAnyOneSpanInList(span, ne_mens_spans):
                    ques_spans_idxs.append(span)

        ques_spans = []
        ques_spans2idx = {}
        ques_spans_linking_score = []
        ques_spans_spanfields: List[SpanField] = []

        for span_idx, (srt, end) in enumerate(ques_spans_idxs):
            tokens = ques_tokens[srt:end]
            span_str = self.make_span_str(span_tokens=tokens, start_idx=srt, end_idx=end-1,
                                          prefix=hpconstants.QSTR_PREFIX)
            if span_str not in ques_spans2idx:
                ques_spans.append(span_str)
                ques_spans2idx[span_str] = len(ques_spans2idx)
                ques_spans_spanfields.append(SpanField(span_start=srt, span_end=end-1,
                                                       sequence_field=ques_textfield))

                linking_score = [0.0]*len(ques_tokens)
                linking_score[srt:end] = [1.0] * (end - srt)
                ques_spans_linking_score.append(linking_score)

        return ques_spans, ques_spans2idx, ques_spans_linking_score, ques_spans_spanfields


    def get_ques_nemens_ent_spans(self,
                                  ques_tokens: List[str],
                                  q_ent_ners: List[Tuple],
                                  q_entmens2entidx: List[int],
                                  num_ne_ents: int,
                                  ques_textfield: TextField,
                                  ques_spans: List[str],
                                  ques_spans2idx: Dict[str, int],
                                  ques_spans_linking_score: List[List[float]],
                                  ques_spans_spanfields: List[SpanField]):
        """ Make question NE mention spans (delimited by SPAN_DELIM)
        Only add spans that are grounded.


        We'll be updating the already existing list/dicts of spans from the QSTR actions
        Add these spans to ques_spans and ques_spans2idx, and append the linking score and SpanField to
        ques_spans_linking_score and ques_spans_spanidxs, resp.

        TODO(nitish): If no NE mens present / no grounded-NE mentions then no actions from QENT -> SPAN will be found
        and will throw an error. Need to check what to do in this case.

        Parameters:
        -----------
        See returns of get_ques_spans function

        Returns:
        --------
        ques_spans: `List[str]`
            All question spans delimited by _
        ques_span2idx: ``Dict[str]: int``
            Dict mapping from q_span to Idx into ques_spans
        ques_spans_linking_score: ``List[List[float]]``
            For each q_span, Binary list the size of q_length indicating q_token's presence in the span
            This is used to score the q_span action in the decoder based on attention
        ques_spans_spanidxs : List[SpanField]
            List of SpanField for each q_span. One use case is to get an output embedding for the decoder.
            Can possibly be used as input embedding as well (will need to figure out the TransitionFunction)
        q_nemenspan2grounding_idx : Dict[str, int]
            Dict from NE_men_span to idx into entity grounding array (q_nemens_grounding)
        # q_nemenspan_grounding : List[List[float]]
        #     For each Q NE Mention, a one-hot vector the size of NEs to indicate the mention grounding
        q_nemenspan2entidx: ``Dict[str, int]``
            Mapping from Q_NE mention span_str to entity_idx (amongst the NE entities in contexts)
        """

        # Ques NE mens span to idx
        q_nemenspan2grounding_idx = {}
        # Q NE Men span to entity_grounding_idx
        q_nemenspan2entidx = {}

        # Ques NE mens span entity grounding -- Each should be a one-hot vector with size as the number of NE entities
        # q_nemenspan_grounding: List[List[float]] = []

        num_mens_added = 0

        for ne_men, entity_grounding in zip(q_ent_ners, q_entmens2entidx):
            if entity_grounding == -1:
                # These are q mens that are not grounded in the contexts
                continue
            span_tokens = ques_tokens[ne_men[1]:ne_men[2]]


            span_str = self.make_span_str(span_tokens=span_tokens, start_idx=ne_men[1], end_idx=ne_men[2] - 1,
                                          prefix=hpconstants.QENT_PREFIX)

            if span_str not in ques_spans2idx:
                ques_spans.append(span_str)
                ques_spans2idx[span_str] = len(ques_spans2idx)

                linkingscore = [0.0 for _ in range(len(ques_tokens))]
                linkingscore[ne_men[1]:ne_men[2]] = [1.0] * (ne_men[2] - ne_men[1])
                ques_spans_linking_score.append(linkingscore)

                ques_spans_spanfields.append(SpanField(span_start=ne_men[1],
                                                       span_end=ne_men[2] - 1,
                                                       sequence_field=ques_textfield))

                q_nemenspan2grounding_idx[span_str] = len(q_nemenspan2grounding_idx)
                q_nemenspan2entidx[span_str] = entity_grounding
                # grounding_vec = [0.0] * num_ne_ents
                # grounding_vec[entity_grounding] = 1.0
                # q_nemenspan_grounding.append(grounding_vec)

                num_mens_added += 1

        # TODO(nitish): Temp solution if a question doesn't have grounded NE mens, make a new random mention
        # Need a better solution than this
        if num_mens_added == 0:
            span_str = self.make_span_str(span_tokens=['RANDOMMENTION'], start_idx=0, end_idx=0,
                                          prefix=hpconstants.QENT_PREFIX)
            ques_spans.append(span_str)
            ques_spans2idx[span_str] = len(ques_spans2idx)
            linkingscore = [0 for _ in range(len(ques_tokens))]
            linkingscore[0] = 1.0
            ques_spans_linking_score.append(linkingscore)
            ques_spans_spanfields.append(SpanField(span_start=0,
                                                   span_end=0,
                                                   sequence_field=ques_textfield))
            q_nemenspan2grounding_idx[span_str] = len(q_nemenspan2grounding_idx)
            # Grounding to arbitrary entity
            q_nemenspan2entidx[span_str] = 0
            # grounding_vec = [0.0] * num_ne_ents
            # q_nemenspan_grounding.append(grounding_vec)

        return (ques_spans, ques_spans2idx, ques_spans_linking_score, ques_spans_spanfields,
                q_nemenspan2grounding_idx, q_nemenspan2entidx)


    @overrides
    def text_to_instance(self,
                         ques: str,
                         q_nemens: List,
                         contexts: List,
                         contexts_ent_ners: List,
                         contexts_num_ners: List,
                         contexts_date_ners: List,
                         context_entmens2entidx: List,
                         context_nummens2entidx: List,
                         context_datemens2entidx: List,
                         context_eqent2entmens: List,
                         context_eqent2nummens: List,
                         context_eqent2datemens: List,
                         dates_normalized_dict: Dict,
                         nums_normalized_dict: Dict,
                         qnemens_to_ent: List,
                         ans_type: str,
                         ans_grounding: Any) -> Instance:
        """
        Parameters
        ----------
        """

        if ans_type not in [hpconstants.BOOL_TYPE]: #, hpconstants.NUM_TYPE]:
            return None

        # pylint: disable=arguments-differ
        tokenized_ques = ques.strip().split(" ")
        tokenized_ques = [hpconstants.COMMA if x == ',' else x for x in tokenized_ques]
        tokenized_ques = [x.replace(',', hpconstants.COMMA) for x in tokenized_ques]

        # Question TextField
        ques_tokens = [Token(token) for token in tokenized_ques]
        ques_tokenized_field = TextField(ques_tokens, self._sentence_token_indexers)

        # Make Ques_spans, their idxs, linking score and SpanField representations
        (ques_spans, ques_spans2idx,
         ques_spans_linking_score, ques_spans_spanfields) = self.get_ques_spans(ques_tokens=tokenized_ques,
                                                                                q_ent_ners=q_nemens,
                                                                                ques_textfield=ques_tokenized_field)
        # Deep copy since ques_spans2idx changes later on
        q_qstr2idx_field = MetadataField(copy.deepcopy(ques_spans2idx))
        q_qstr_spanfield = ListField(ques_spans_spanfields)

        num_ne_ents = len(context_eqent2entmens)

        # Processing for NE mens in the question
        # ques_spans, ques_spans2idx, ques_spans_linking_score, ques_spans_spanidxs - updated with the NE mens
        # q_nemenspan2grounding_idx, q_nemenspan_grounding - Second contains grounding vector, first is an idx into it
        (ques_spans, ques_spans2idx,
         ques_spans_linking_score, ques_spans_spanidxs,
         q_nemenspan2grounding_idx, q_nemenspan2entidx) = self.get_ques_nemens_ent_spans(
            ques_tokens=tokenized_ques, q_ent_ners=q_nemens, q_entmens2entidx=qnemens_to_ent,
            num_ne_ents=num_ne_ents, ques_textfield=ques_tokenized_field,
            ques_spans=ques_spans, ques_spans2idx=ques_spans2idx,
            ques_spans_linking_score=ques_spans_linking_score, ques_spans_spanfields=ques_spans_spanfields)

        q_nemens2groundingidx_field = MetadataField(q_nemenspan2grounding_idx)
        # q_nemens_grounding_field = ArrayField(np.array(q_nemenspan_grounding), padding_value=-1)
        q_nemenspan2entidx_field = MetadataField(q_nemenspan2entidx)

        # Make world from question spans
        # This adds instance specific actions as well
        # world = SampleHotpotWorld(ques_spans=ques_spans)
        hplanguage = HotpotQALanguage(qstr_qent_spans=ques_spans)
        lang_field = MetadataField(hplanguage)

        # Action_field:
        #   Currently, all instance-specific rules are terminal rules of the kind: q -> QSTR:ques_span
        # Action_linking_scores:
        #   Create a dictionary mapping, linked_rule2idx: {linked_rule: int_idx}
        #   Create a DataArray of linking scores: (num_linked_rules, num_ques_tokens)
        #   With the linked_rule2idx map, the correct linking_score can be retrieved in the world.
        production_rule_fields: List[ProductionRuleField] = []
        linked_rule2idx = {}
        action_to_ques_linking_scores = ques_spans_linking_score
        for production_rule in hplanguage.all_possible_productions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not (rule_right_side.startswith(hpconstants.QSTR_PREFIX)
                                  or rule_right_side.startswith(hpconstants.QENT_PREFIX))
            rule_field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(rule_field)

            # Tokens in a ques_span; rule_right_side is QSTR_PREFIXquestion_span, hence removing the QSTR_PREFIX
            if not is_global_rule:
                # ques_span = rule_right_side[len(QSTR_PREFIX):]
                linked_rule2idx[production_rule] = ques_spans2idx[rule_right_side]

        action_field = ListField(production_rule_fields)
        # Dict from linked_rule_string -> idx in 'action_to_ques_linking_scores'
        linked_rule2idx_field = MetadataField(linked_rule2idx)
        action_to_ques_linking_scores_field = ArrayField(np.array(action_to_ques_linking_scores), padding_value=0)
        ques_span_actions_to_spanfield = ListField(ques_spans_spanidxs)

        # print(f"World actions: {world.all_possible_actions()}")
        # print(f"DatasetReader actions: {production_rule_fields}")

        contexts_tokenized = []
        for context_id, context in contexts:
            tokenized_context = context.strip().split(" ")
            tokenized_context = [Token(token) for token in tokenized_context]
            tokenized_context = TextField(tokenized_context, self._sentence_token_indexers)
            contexts_tokenized.append(tokenized_context)
        contexts_tokenized_field = ListField(contexts_tokenized)

        # all_ent_mens = self.processEntMens(contexts_ent_ners, contexts_tokenized)
        all_ent_mens, _ = self.processMens(mens=contexts_ent_ners, eqent2mens=context_eqent2entmens,
                                           contexts_tokenized=contexts_tokenized)
        all_ent_mens_field = ListField(all_ent_mens)

        # For each context, List of NUM-type men spans
        all_num_mens, all_nummens_normval = self.processMens(mens=contexts_num_ners, eqent2mens=context_eqent2nummens,
                                                             contexts_tokenized=contexts_tokenized,
                                                             normalization_dict=nums_normalized_dict,
                                                             FieldClass=NumberField)
        all_num_mens_field = ListField(all_num_mens)
        all_nummens_normval_field = ListField(all_nummens_normval)

        all_date_mens, all_datemens_normval = self.processMens(mens=contexts_date_ners,
                                                               eqent2mens=context_eqent2datemens,
                                                               contexts_tokenized=contexts_tokenized,
                                                               normalization_dict=dates_normalized_dict,
                                                               FieldClass=DateField)
        all_date_mens_field = ListField(all_date_mens)
        all_datemens_normval_field = ListField(all_datemens_normval)

        (num_nents, num_numents, num_dateents) = (ArrayField(np.array([len(context_eqent2entmens)])),
                                                  ArrayField(np.array([len(context_eqent2nummens)])),
                                                  ArrayField(np.array([len(context_eqent2datemens)])))

        fields: Dict[str, Field] = {"question": ques_tokenized_field,
                                    "q_qstr2idx": q_qstr2idx_field,
                                    "q_qstr_spans": q_qstr_spanfield,
                                    "q_nemens2groundingidx": q_nemens2groundingidx_field,
                                    "q_nemenspan2entidx": q_nemenspan2entidx_field,
                                    "contexts": contexts_tokenized_field,
                                    "ent_mens": all_ent_mens_field,
                                    "num_nents": num_nents,
                                    "num_mens": all_num_mens_field,
                                    "num_numents": num_numents,
                                    "date_mens": all_date_mens_field,
                                    "num_dateents": num_dateents,
                                    "num_normval": all_nummens_normval_field,
                                    "date_normval": all_datemens_normval_field,
                                    "languages": lang_field,
                                    "actions": action_field,
                                    "linked_rule2idx": linked_rule2idx_field,
                                    "action2ques_linkingscore": action_to_ques_linking_scores_field,
                                    "ques_str_action_spans": ques_span_actions_to_spanfield
                                    }

        # TODO(nitish): Figure out how to pack the answer. Multiple types; ENT, BOOL, NUM, DATE, STRING
        # One way is to have field for all types of answer, and mask all but the correct kind.

        # booltype_ans_prob, enttype_ans_prob, numtype_ans_prob, \
        # datetype_ans_prob, strtype_ans_prob = 0.0, 0.0, 0.0, 0.0, 0.0


        # Answers can have multiple types. So grounding might be tricky.
        # ENTITY_TYPE, NUM_TYPE, DATE_TYPE:
        # BOOL_TYPE:
        # STRING_TYPE:

        num_enttype_ents = len(context_eqent2entmens)
        num_numtype_ents = len(context_eqent2nummens)
        num_datetype_ents = len(context_eqent2datemens)

        if ans_grounding is not None:

            # If the true answer type is STRING, convert the grounding in a allennlp consumable repr.
            if ans_type == hpconstants.STRING_TYPE:
                if ans_grounding == hpconstants.NO_ANS_GROUNDING:
                    return None
                # Returns: List[ListField[SpanField]]
                ans_grounding = self.processStringGrounding(num_contexts=len(contexts),
                                                            preprocessed_string_grounding=ans_grounding,
                                                            contexts_tokenized=contexts_tokenized)
            if ans_type == hpconstants.BOOL_TYPE:
                ans_grounding = [ans_grounding]

            # Make a Dict with keys as Type and value is an empty ans grounding for them.
            # The answer grounding repr depends on the type
            empty_ans_groundings = self.emptyAnsGroundings(num_contexts=len(contexts),
                                                           num_enttype_ents=num_enttype_ents,
                                                           num_num_ents=num_numtype_ents,
                                                           num_date_ents=num_datetype_ents,
                                                           contexts_tokenized=contexts_tokenized)




            # For the correct asn type, replace the empty ans_grounding with the ground_truth
            ans_grounding_dict = empty_ans_groundings
            ans_grounding_dict[ans_type] = ans_grounding

            for k, v in ans_grounding_dict.items():
                if k != hpconstants.STRING_TYPE:
                    fields["ans_grounding_" + k] = ArrayField(np.array(v), padding_value=-1)
                else:
                    fields["ans_grounding_" + k] = ListField(v)

            ans_type_field =  MetadataField(ans_type) # LabelField(ans_type, label_namespace="anstype_labels")
            fields["gold_ans_type"] = ans_type_field

        return Instance(fields)

    def emptyAnsGroundings(self, num_contexts, num_enttype_ents, num_num_ents, num_date_ents, contexts_tokenized):
        bool_grounding = [0.0]

        # Empty span for each context
        string_grounding = [[SpanField(-1, -1, contexts_tokenized[i])] for i in range(0, num_contexts)]
        string_grounding = [ListField(x) for x in string_grounding]

        # Entity-type grounding
        ent_grounding = [0] * num_enttype_ents

        # Number-type grounding
        num_grounding = [0] * num_num_ents

        # Date-type grounding
        date_grounding = [0] * num_date_ents

        return {hpconstants.BOOL_TYPE: bool_grounding,
                hpconstants.STRING_TYPE: string_grounding,
                hpconstants.ENTITY_TYPE: ent_grounding,
                hpconstants.DATE_TYPE: date_grounding,
                hpconstants.NUM_TYPE: num_grounding}

    def processStringGrounding(self, num_contexts, preprocessed_string_grounding, contexts_tokenized):
        '''String ans grounding is gives as: List (context_idx, (start, end)).
            Convert this into a List of list of spans, i.e. for each context, a list of spans in it.
            Technically, a List of ListField of SpanField
        '''

        allcontexts_ans_spans = [[] for context_idx in range(0, num_contexts)]

        for (context_idx, (start, end)) in preprocessed_string_grounding:
            # Making the end inclusive
            allcontexts_ans_spans[context_idx].append(SpanField(start, end - 1, contexts_tokenized[context_idx]))

        for context_idx, context_spans in enumerate(allcontexts_ans_spans):
            if len(context_spans) == 0:
                allcontexts_ans_spans[context_idx].append(SpanField(-1, - 1, contexts_tokenized[context_idx]))

        for context_idx in range(0, len(allcontexts_ans_spans)):
            allcontexts_ans_spans[context_idx] = ListField(allcontexts_ans_spans[context_idx])

        return allcontexts_ans_spans

    def processMens(self, mens, eqent2mens, contexts_tokenized,
                    normalization_dict=None, FieldClass=None):
        """ Process date mentions and entities to make Mention datatypes and equivalent normalized values for pytorch

        Parameters:
        -----------
        mens: ``List[List[NER]]``
            For each context, list of date mentions. Each mention is a (menstr, start, end, type) tuple. Exclusive end
        eqent2emens: ``List[List[(context_idx, mention_idx)]]``
            For each entity, list of mentions referring to it. Mentions are (context_idx, mention_idx) tuples
        normalization_dict: ``Dict``
            Dict from {mention_str: normalized_val}.
        contexts_tokenized: ``List[TextField]``
            List of text datatypes of contexts to make SapnFields

        Returns:
         -------
        """
        num_contexts = len(contexts_tokenized)

        AnyFC = TypeVar('FieldClass', DateField, NumberField)
        FieldClass: AnyFC = FieldClass


        # Make a E, C, M list field
        # i.e For each entity, for each context, make a ListField of Spans wrapped into a Listfield.
        all_entity_mentions = []
        entity_normalizations = None
        if normalization_dict is not None:
            entity_normalizations = []

        if len(eqent2mens) == 0:
            entity_mentions = [ListField([ListField([SpanField(-1, -1, contexts_tokenized[i])]) for i in range(num_contexts)])]
            if normalization_dict is not None:
                entity_normalizations = [FieldClass.empty_object()]

            return entity_mentions, entity_normalizations

        for e_idx, entity in enumerate(eqent2mens):
            entity_mentions = [[] for i in range(num_contexts)]
            entity_normval = None
            # Each entity is a list of (context_idx, mention_idx)
            for (c_idx, m_idx) in entity:
                mention_tuple = mens[c_idx][m_idx]
                context_textfield = contexts_tokenized[c_idx]
                start, end = mention_tuple[1], mention_tuple[2] - 1
                if normalization_dict is not None:
                    if entity_normval is None:
                        entity_normval = normalization_dict[mention_tuple[0]]
                    else:
                        assert entity_normval == normalization_dict[mention_tuple[0]]
                mention_field = SpanField(span_start=start, span_end=end, sequence_field=context_textfield)
                entity_mentions[c_idx].append(mention_field)
            single_entity_mentions_field = []

            # For each context, convert the list to a ListField.
            # For contexts with no mention of this entity, add an empty SpanField
            for c_idx, c_mens in enumerate(entity_mentions):
                if len(c_mens) == 0:
                    empty_men_field = SpanField(span_start=-1, span_end=-1, sequence_field=contexts_tokenized[c_idx])
                    entity_mentions[c_idx].append(empty_men_field)
                single_entity_mentions_field.append(ListField(entity_mentions[c_idx]))

            all_entity_mentions.append(ListField(single_entity_mentions_field))

            # entity_normval = MetadataField(entity_normval)
            if normalization_dict is not None:
                assert entity_normval is not None, f"Normalization of entity shouldn't be None: {mens}"
                entity_normval = FieldClass(entity_normval)
                entity_normalizations.append(entity_normval)
            else:
                entity_normalizations = None

        return all_entity_mentions, entity_normalizations

    # def processEntMens(self, ent_mens, contexts_tokenized):
    #     all_ent_mens = []
    #     for context_idx, context in enumerate(ent_mens):
    #         ent_men_spans = []
    #         for men in context:
    #             ent_men_spans.append(SpanField(men[1], men[2] - 1, contexts_tokenized[context_idx]))
    #         if len(ent_men_spans) == 0:
    #             ent_men_spans.append(SpanField(-1, -1, contexts_tokenized[context_idx]))
    #
    #         ent_men_spans = ListField(ent_men_spans)
    #         all_ent_mens.append(ent_men_spans)
    #
    #     return all_ent_mens

