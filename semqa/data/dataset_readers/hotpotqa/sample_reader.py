from typing import Dict, List, Any
import json
import logging

import numpy as np

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField
from allennlp.data.fields import ProductionRuleField, MetadataField, SpanField, ArrayField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# from allennlp.semparse.worlds import NlvrWorld
from semqa.worlds.hotpotqa.sample_world import SampleHotpotWorld
import datasets.hotpotqa.utils.constants as hpconstants

from allennlp.data.dataset_readers.wikitables import WikiTablesDatasetReader


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

QSTR_PREFIX="QSTR:"


@DatasetReader.register("sample_hotpot")
class SampleHotpotDatasetReader(DatasetReader):
    """
    Copied from NlvrDatasetReader
    (https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/semantic_parsing/nlvr.py)
    """

    """
    ``DatasetReader`` for the NLVR domain. In addition to the usual methods for reading files and
    instances from text, this class contains a method for creating an agenda of actions that each
    sentence triggers, if needed. Note that we deal with the version of the dataset with structured
    representations of the synthetic images instead of the actual images themselves.

    We support multiple data formats here:
    1) The original json version of the NLVR dataset (http://lic.nlp.cornell.edu/nlvr/) where the
    format of each line in the jsonl file is
    ```
    "sentence": <sentence>,
    "label": <true/false>,
    "identifier": <id>,
    "evals": <dict containing all annotations>,
    "structured_rep": <list of three box representations, where each box is a list of object
    representation dicts, containing fields "x_loc", "y_loc", "color", "type", "size">
    ```

    2) A grouped version (constructed using ``scripts/nlvr/group_nlvr_worlds.py``) where we group
    all the worlds that a sentence appears in. We use the fields ``sentence``, ``label`` and
    ``structured_rep``.  And the format of the grouped files is
    ```
    "sentence": <sentence>,
    "labels": <list of labels corresponding to worlds the sentence appears in>
    "identifier": <id that is only the prefix from the original data>
    "worlds": <list of structured representations>
    ```

    3) A processed version that contains action sequences that lead to the correct denotations (or
    not), using some search. This format is very similar to the grouped format, and has the
    following extra field

    ```
    "correct_sequences": <list of lists of action sequences corresponding to logical forms that
    evaluate to the correct denotations>
    ```

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
                qners = data[hpconstants.q_ner_field]
                # List of (title, space_delimited_tokenized_contexts)
                contexts = data[hpconstants.context_field]
                # List of list --- For each context , list of mention-tuples as (text, start, end, label)
                context_ners = data[hpconstants.context_ner_field]
                # Dict from {date_string: (date, month, year)} normalization. -1 indicates invalid field
                dates_normalized_dict = data[hpconstants.dates_normalized_field]
                # Dict from {num_string: float_val} normalization.
                nums_normalized_dict = data[hpconstants.nums_normalized_field]
                # Dict from {ent_idx: [(context_idx, men_idx)]} --- output pf CDCR
                ent_to_mens = data[hpconstants.ENT_TO_CONTEXT_MENS]
                # List of list --- For each context , for each mention the ent_idx it is grounded to
                # For DATE or NUM type, the grounding is constants.NOLINK
                mens_to_ent = data[hpconstants.CONTEXT_MENS_TO_ENT]
                # (list) grounding for qmens, constants.NOLINK for qmen that cannot be grounded
                qmens_to_ent = data[hpconstants.Q_MENS_TO_ENT]

                ans_type = None
                ans_grounding = None
                if hpconstants.ans_type_field in data:
                    ans_type = data[hpconstants.ans_type_field]
                    ans_grounding = data[hpconstants.ans_grounding_field]

                instance = self.text_to_instance(question,
                                                 qners,
                                                 contexts,
                                                 context_ners,
                                                 dates_normalized_dict,
                                                 nums_normalized_dict,
                                                 ent_to_mens,
                                                 mens_to_ent,
                                                 qmens_to_ent,
                                                 ans_type,
                                                 ans_grounding)
                if instance is not None:
                    yield instance

    def get_ques_spans(self, ques_tokens: List[str], ques_textfield: TextField):
        """ Make question spans (delimited by _ ) and also return their spans (inclusive)
        Current implementation: Only take tokens as spans.

        TODO(nitish): How to represent spans that occur multiple times in a question.
                      Current solution: Only take the first occurrence span
        TODO(nitish): 1) Extend to longer spans, 2) Don't break entity spans

        Parameters:
        -----------
        ques_tokens: List[str]: List of question tokens
        ques_textfield: TextField: For SpanFields

        Returns:
        --------
        ques_spans: `List[str]` All question spans delimited by _
        ques_span2idx:
        ques_spans_linking_score:
        ques_spans_spanidxs:
        """

        # Will be used for linking scores
        qtoken2idxs = {}
        for qtokenidx, qtoken in enumerate(ques_tokens):
            if qtoken not in qtoken2idxs:
                qtoken2idxs[qtoken] = []
            qtoken2idxs[qtoken].append(qtokenidx)

        uniq_ques_tokens = list(set(ques_tokens))

        ques_spans = []
        ques_spans2idx = {}
        ques_spans_linking_score = []
        ques_spans_spanidxs: List[SpanField] = []



        for tokenidx, token in enumerate(uniq_ques_tokens):
            span = token
            # span --- will be _ delimited later on
            if span not in ques_spans2idx:
                ques_spans.append(token)
                ques_spans2idx[span] = len(ques_spans2idx)
                ques_spans_spanidxs.append(SpanField(span_start=tokenidx, span_end=tokenidx,
                                                     sequence_field=ques_textfield))

                span_tokens = span.split("_")
                linking_score = [0.0]*len(ques_tokens)
                for token in span_tokens:
                    # Single token can occur multiple times in the question
                    for idx in qtoken2idxs[token]:
                        linking_score[idx] = 1.0
                ques_spans_linking_score.append(linking_score)

        return (ques_spans, ques_spans2idx, ques_spans_linking_score, ques_spans_spanidxs)

    @overrides
    def text_to_instance(self,
                         ques: str,
                         qners: List,
                         contexts: List,
                         context_ners: List,
                         dates_normalized_dict: Dict,
                         nums_normalized_dict: Dict,
                         ent_to_mens: Dict,
                         mens_to_ent: List,
                         qmens_to_ent: List,
                         ans_type: str,
                         ans_grounding: Any) -> Instance:
        """
        Parameters
        ----------
        """

        if ans_type != hpconstants.BOOL_TYPE:
            return None

        # pylint: disable=arguments-differ
        tokenized_ques = ques.strip().split(" ")

        ques_tokens = [Token(token) for token in tokenized_ques]
        ques_tokenized_field = TextField(ques_tokens, self._sentence_token_indexers)

        (ques_spans, ques_spans2idx,
         ques_spans_linking_score, ques_spans_spanidxs) = self.get_ques_spans(ques_tokens=tokenized_ques,
                                                                              ques_textfield=ques_tokenized_field)

        world = SampleHotpotWorld(ques_spans=ques_spans)
        world_field = MetadataField(world)

        # Action_field:
        #   Currently, all instance-specific rules are terminal rules of the kind: q -> QSTR:ques_sub_span
        # Action_linking_scores:
        #   Create a dictionary mapping, linked_rule2idx: {linked_rule: int_idx}
        #   Create a DataArray of linking scores: (num_linked_rules, num_ques_tokens)
        #   With the linked_rule2idx map, the correct linking_score can be retrieved in the world.
        production_rule_fields: List[Field] = []
        linked_rule2idx = {}
        action_to_ques_linking_scores = ques_spans_linking_score
        for production_rule in world.all_possible_actions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not rule_right_side.startswith(QSTR_PREFIX)
            rule_field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(rule_field)

            # Tokens in a ques_span; rule_right_side is QSTR_PREFIXquestion_span, hence removing the QSTR_PREFIX
            if not is_global_rule:
                ques_span = rule_right_side[len(QSTR_PREFIX):]
                linked_rule2idx[production_rule] = ques_spans2idx[ques_span]

        action_field = ListField(production_rule_fields)
        linked_rule2idx_field = MetadataField(linked_rule2idx)
        action_to_ques_linking_scores_field = ArrayField(np.array(action_to_ques_linking_scores), padding_value=0)
        action_to_span_field = ListField(ques_spans_spanidxs)

        # print(f"World actions: {world.all_possible_actions()}")
        # print(f"DatasetReader actions: {production_rule_fields}")

        contexts_tokenized = []
        for context_id, context in contexts:
            tokenized_context = context.strip().split(" ")
            tokenized_context = [Token(token) for token in tokenized_context]
            tokenized_context = TextField(tokenized_context, self._sentence_token_indexers)
            contexts_tokenized.append(tokenized_context)
        contexts_tokenized_field = ListField(contexts_tokenized)

        # For each context, List of NUM-type men spans
        all_num_mens = []
        all_nummens_normval = []
        for context_idx, context in enumerate(context_ners):
            num_men_spans = []
            num_men_normval = []
            for men in context:
                if men[-1] == hpconstants.NUM_TYPE:
                    num_men_spans.append(SpanField(men[1], men[2] - 1, contexts_tokenized[context_idx]))
                    num_men_normval.append(MetadataField(nums_normalized_dict[men[0]]))
            if len(num_men_spans) == 0:
                num_men_spans.append(SpanField(-1, -1, contexts_tokenized[context_idx]))
                num_men_normval.append(MetadataField(-1))

            num_men_spans = ListField(num_men_spans)
            num_men_normval = ListField(num_men_normval)
            all_num_mens.append(num_men_spans)
            all_nummens_normval.append(num_men_normval)
        all_num_mens_field = ListField(all_num_mens)
        all_nummens_normval_field = ListField(all_nummens_normval)

        fields: Dict[str, Field] = {"question": ques_tokenized_field,
                                    "contexts": contexts_tokenized_field,
                                    "num_mens_field": all_num_mens_field,
                                    "num_normval_field": all_nummens_normval_field,
                                    "worlds": world_field,
                                    "actions": action_field,
                                    "linked_rule2idx": linked_rule2idx_field,
                                    "action2ques_linkingscore": action_to_ques_linking_scores_field,
                                    "action2span": action_to_span_field
                                    }

        # TODO(nitish): Figure out how to pack the answer. Multiple types; ENT, BOOL, NUM, DATE, STRING
        # One way is to have field for all types of answer, and mask all but the correct kind.

        # booltype_ans_prob, enttype_ans_prob, numtype_ans_prob, \
        # datetype_ans_prob, strtype_ans_prob = 0.0, 0.0, 0.0, 0.0, 0.0


        # Answers can have multiple types. So grounding might be tricky.
        # ENTITY_TYPE, NUM_TYPE, DATE_TYPE:
        # BOOL_TYPE:
        # STRING_TYPE:

        ans_type_field = LabelField(ans_type, label_namespace="anstype")

        if ans_grounding is not None:
            # Currently only dealing with boolean answers
            ans_field = ArrayField(np.array([ans_grounding]))
            fields["ans_grounding"] = ans_field
            fields["ans_type"] = ans_type_field

        '''
        # Depending on the type of supervision used for training the parser, we may want either
        # target action sequences or an agenda in our instance. We check if target sequences are
        # provided, and include them if they are. If not, we'll get an agenda for the sentence, and
        # include that in the instance.
        if target_sequences:
            action_sequence_fields: List[Field] = []
            for target_sequence in target_sequences:
                index_fields = ListField([IndexField(instance_action_ids[action], action_field)
                                          for action in target_sequence])
                action_sequence_fields.append(index_fields)
                # TODO(pradeep): Define a max length for this field.
            fields["target_action_sequences"] = ListField(action_sequence_fields)
        elif self._output_agendas:
            # TODO(pradeep): Assuming every world gives the same agenda for a sentence. This is true
            # now, but may change later too.
            agenda = worlds[0].get_agenda_for_sentence(sentence, add_paths_to_agenda=False)
            assert agenda, "No agenda found for sentence: %s" % sentence
            # agenda_field contains indices into actions.
            agenda_field = ListField([IndexField(instance_action_ids[action], action_field)
                                      for action in agenda])
            fields["agenda"] = agenda_field
        if labels:
            labels_field = ListField([LabelField(label, label_namespace='denotations')
                                      for label in labels])
            fields["labels"] = labels_field
        '''

        return Instance(fields)