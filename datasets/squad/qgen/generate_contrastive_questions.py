from typing import List, Tuple, Dict, Union, Callable
import os
import json
import argparse
import networkx as nx

from allennlp.predictors import Predictor
from allennlp.data.tokenizers import SpacyTokenizer

from utils import util, spacyutils
from datasets.drop import constants
from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    read_json_dataset, read_jsonl
from semqa.predictors.qgen_predictor import QuestionGenerationPredictor
from semqa.models.qgen.conditional_qgen_model import ConditionalQuestionGenerationModel
from semqa.data.dataset_readers.qgen.squad_qgen_reader import SquadConditionalQuestionGenerationReader
from semqa.predictors.constituency_parser import ConstituencyParserPredictor
from datasets.squad.squad_utils import make_qa_pair_dict, add_project_select_program_supervision


nlp_spacy = spacyutils.getSpacyNLP()
spacy_tokenizer = SpacyTokenizer()


Entity = Tuple[int, int, str]
CharOffsets = Tuple[int, int]

def get_question_generation_predictor(model_tar_gz) -> QuestionGenerationPredictor:
    print("Loading QGen model")
    predictor = Predictor.from_path(archive_path=model_tar_gz, cuda_device=0, predictor_name="question_generation")
    return predictor


def get_constituency_parser_predictor(model_tar_gz) -> ConstituencyParserPredictor:
    model_tar_gz = "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
    predictor = Predictor.from_path(archive_path=model_tar_gz, cuda_device=0, predictor_name="my-constituency-parser")
    return predictor


def get_sent_idx(sent_charoffsets: List[CharOffsets], span_charoffset: CharOffsets):
    """Sentence charoffsets are sorted in decreasing order"""
    span_start, span_end = span_charoffset

    if span_start > sent_charoffsets[-1][-1]:
        print("Span outside document")
        return len(sent_charoffsets) - 1

    for sent_idx, (sent_start, sent_end) in enumerate(sent_charoffsets):
        if util.isSpanOverlap((sent_start, sent_end), span_charoffset, exclusive=True):
            return sent_idx
    # Answer not overlapping with a sentence, weird
    return None


def get_entities_in_sent(sent_charoffset: CharOffsets, entities: List[Entity]):
    """Get all entities within the sentence. """

    relevant_entities = []
    for (start, end, label) in entities:
        if start >= sent_charoffset[0] and end <= sent_charoffset[1]:
            relevant_entities.append((start, end, label))
    return relevant_entities


def remove_overlapping_entities(entities: List[Entity], answer_offsets: CharOffsets):
    """Return entities that do not overlap with the answer.

    Entities are sorted in the order of their appearance.
    """
    relevant_entities = []
    for entity in entities:
        (start, end, label) = entity
        if not util.isSpanOverlap((start, end), answer_offsets, exclusive=False):
            relevant_entities.append(entity)
    return relevant_entities


def sort_entities_by_distance_to_ans(entities: List[Entity], answer_offsets: CharOffsets):
    """Sort entities (non-overlapping w/ answer) based on their distance to the answer. """
    ent_distances = []
    ans_start, ans_end = answer_offsets
    for entity in entities:
        (start, end, label) = entity
        if end < ans_start:
            # entity is before the answer
            ent_distances.append(ans_start - end)
        else:
            # entity has to start after answer ends.
            ent_distances.append(start - ans_end)

    # List of (idx, distance) in sorted manner
    sorted_distances = sorted(enumerate(ent_distances), key=lambda x:x[1])
    sorted_entites = [entities[idx] for idx, _ in sorted_distances]
    return sorted_entites


def get_contrastive_question_program(ques_tokens, ques_spans):
    relevant_span_labels = ["WHNP", "WHADVP", "WP", "WHPP", "WRB", "WDT", "WHADJP"]
    first_span = ques_spans[1]  # skipping full sentence span

    project_start_idx, project_end_idx = None, None
    select_start_idx, select_end_idx = None, None

    if first_span[3] in relevant_span_labels and first_span[0] == 0:
        project_start_idx = first_span[0]
        project_end_idx = first_span[1] - 1

    # for span in ques_parse:
    #     if span[3] in relevant_span_labels and span[0] == 0:
    #         project_start_idx = span[0]
    #         project_end_idx = span[1] - 1  # _inclusive_
    #         break

    if project_start_idx is not None:
        select_start_idx = project_end_idx + 1
        select_end_idx = len(ques_tokens) - 2  # skip last token == ?
    else:
        return None
    project_start_idx: int = project_start_idx
    project_end_idx: int = project_end_idx
    select_start_idx: int = select_start_idx
    select_end_idx: int = select_end_idx

    project_node = Node(predicate="project_passage",
                        string_arg=" ".join(ques_tokens[project_start_idx:project_end_idx + 1]))
    project_ques_attention = [1 if project_start_idx <= i <= project_end_idx else 0 for i in range(len(ques_tokens))]
    project_node.supervision["question_attention_supervision"] = project_ques_attention

    select_node = Node(predicate="select_passage",
                       string_arg=" ".join(ques_tokens[select_start_idx:select_end_idx + 1]))
    select_ques_attention = [1 if select_start_idx <= i <= select_end_idx else 0 for i in range(len(ques_tokens))]
    select_node.supervision["question_attention_supervision"] = select_ques_attention

    project_node.add_child(select_node)

    program_node = Node(predicate="select_passagespan_answer")
    program_node.add_child(project_node)

    program_node.extras["constituency_parse_spans"] = ques_spans
    program_node.extras["project_start_end_indices"] = (project_start_idx, project_end_idx)
    program_node.extras["select_start_end_indices"] = (select_start_idx, select_end_idx)

    return program_node



def get_substructure_annotation_dict(aux_question: str,
                                     aux_question_tokens: List[str],
                                     aux_program_node: Node,
                                     orig_program_lisp: str,
                                     orig_question: str,
                                     origprog_postorder_node_idx: int,
                                     sharedprog_postorder_node_idx: int) -> Dict:
    # A list of such dicts should be added to qa[constants.shared_substructure_annotations]
    return {
        constants.question: aux_question,
        constants.question_tokens: aux_question_tokens,
        constants.program_supervision: aux_program_node.to_dict(),
        "orig_program_lisp": orig_program_lisp,
        "orig_question": orig_question,
        "origprog_postorder_node_idx": origprog_postorder_node_idx,
        "sharedprog_postorder_node_idx": sharedprog_postorder_node_idx,
    }


def get_sentence_graph(sent):
    # sent_start_idx = sent.start
    edges = []
    for token in sent:
        tokenidx = token.i # - sent_start_idx
        for child in token.children:
            childidx = child.i # - sent_start_idx
            edges.append(('{}'.format(tokenidx),
                          '{}'.format(childidx)))
    graph = nx.Graph(edges)
    return graph


def get_dep_distance_between_entities(sent_graph, ent1_tokens: List[int], ent2_tokens: List[int]):
    """Get min dependency-distance between two entities.

    Entity token idxs are global document idxs
    """
    min_distance = 1000000
    for idx1 in ent1_tokens:
        for idx2 in ent2_tokens:
            shortest_path = shortest_dependency_path(sent_graph, idx1, idx2)
            if shortest_path is None:
                distance = 1000000
            else:
                distance = len(shortest_path)
            min_distance = min(min_distance, distance)
    return min_distance


def shortest_dependency_path(graph, e1: int, e2: int):
    try:
        shortest_path = nx.shortest_path(graph, source=str(e1), target=str(e2))
    except:
        shortest_path = None
    return shortest_path


def get_token_idx(charoffset, spacy_doc):
    tokenidx = None
    for tidx, token in enumerate(spacy_doc):
        # charoffset is inclusive; so match starting from first char
        if token.idx <= charoffset < token.idx + len(token.text):
            return tidx

    return None


def get_contrastive_questions(squad_dataset: Dict, qgen_model_targz):
    qgen_predictor: QuestionGenerationPredictor = get_question_generation_predictor(qgen_model_targz)
    conparse_predictor: ConstituencyParserPredictor = get_constituency_parser_predictor(None)

    total_q = 0
    q_w_pairedq = 0

    total_paras = 0
    # new_sample_dataset = {}

    for passage_id, passage_info in squad_dataset.items():
        passage = passage_info[constants.passage]
        passage_tokens = passage_info[constants.passage_tokens]
        # For each sentence (start, end _exclusive) token offsets
        passage_sentidxs: List[Tuple[int, int]] = passage_info[constants.passage_sent_idxs]
        # For each token, start charoffset
        passage_charoffsets: List[int] = passage_info[constants.passage_charidxs]

        # (start, end) _exclusive_ Char offsets for sentences
        sent_charoffsets = [(passage_charoffsets[sent_start],
                             passage_charoffsets[sent_end-1] + len(passage_tokens[sent_end-1]))
                            for (sent_start, sent_end) in passage_sentidxs]
        sent_charoffsets = sorted(sent_charoffsets, key=lambda x: x[0])

        passage_spacydoc = nlp_spacy(passage)

        # List of (start-tokenoffset, end-tokenoffset (_inclusive), label)
        entities: List[Entity] = []
        for ent in passage_spacydoc.ents:
            start, end = ent.start, ent.end - 1
            # start_charoffset = passage_charoffsets[start]
            # end_charoffset = passage_charoffsets[end] + len(passage_tokens[end])
            label = ent.label_
            entities.append((start, end, label))

        entities = sorted(entities, key=lambda x:x[0])

        for qa in passage_info[constants.qa_pairs]:
            question = qa[constants.question]
            qid = qa[constants.query_id]
            progsupervision = qa.get(constants.program_supervision, None)
            total_q += 1
            if total_q % 500 == 0:
                print("questions processed: {}".format(total_q))

            if progsupervision is None:
                # Add contrastive questions only for questions with prog-supervision
                continue

            answer_dict = qa[constants.answer]
            answer_text = answer_dict["spans"][0]  # SQuAD only has a single span answer
            if not answer_text:
                continue

            answer_start_charoffsets = list(util._KnuthMorrisPratt(passage, answer_text))
            ans_start_offset, ans_end_offset = answer_start_charoffsets[0], answer_start_charoffsets[0] + len(
                answer_text)

            answer_start_tokenidx = get_token_idx(ans_start_offset, passage_spacydoc)
            answer_end_tokenidx = get_token_idx(ans_end_offset - 1, passage_spacydoc)

            if answer_start_tokenidx is None or answer_end_tokenidx is None:
                continue

            answer_offsets = (answer_start_tokenidx, answer_end_tokenidx)

            sentence = passage_spacydoc[answer_start_tokenidx].sent
            relevant_entities = []
            for ent in sentence.ents:
                start, end = ent.start, ent.end - 1
                label = ent.label_
                relevant_entities.append((start, end, label))

            relevant_entities = remove_overlapping_entities(relevant_entities, answer_offsets)
            if not relevant_entities:
                # If no NEs in the sentence containing the answer
                continue

            # Sort entities by token distance; helps if dependency distances clash
            relevant_entities_sorted = sort_entities_by_distance_to_ans(relevant_entities, answer_offsets)
            sentence_graph = get_sentence_graph(sentence)
            entity_dependency_distances = []
            for entity in relevant_entities_sorted:
                distance = get_dep_distance_between_entities(sentence_graph,
                                                             range(answer_start_tokenidx,answer_end_tokenidx+1),
                                                             range(entity[0], entity[1] + 1))
                entity_dependency_distances.append(distance)
            shortest_distance = min(entity_dependency_distances)
            nearest_entity_idx = entity_dependency_distances.index(shortest_distance)
            nearest_entity = relevant_entities_sorted[nearest_entity_idx]

            nearest_entity_start_tokenidx, nearest_entity_end_tokenidx = nearest_entity[0], nearest_entity[1]
            nearest_entity_start_charoffset = passage_spacydoc[nearest_entity_start_tokenidx].idx
            nearest_entity_end_charoffset = (passage_spacydoc[nearest_entity_end_tokenidx].idx +
                                             len(passage_spacydoc[nearest_entity_end_tokenidx].text))

            # Taking the first entity near the answer as contrastive-answer
            contrastive_answer_offsets = [nearest_entity_start_charoffset, nearest_entity_end_charoffset]
            contrastive_answer_text = passage[nearest_entity_start_charoffset:nearest_entity_end_charoffset]

            qgen_output = qgen_predictor.predict(passage=passage,
                                                 answer_text=contrastive_answer_text,
                                                 answer_start_charoffsets=[contrastive_answer_offsets[0]])
            contrastive_question = qgen_output['predicted_question']

            contrastive_qa_dict = make_qa_pair_dict(qid=qid+"-contrastive-1", question=contrastive_question,
                                                    answer_texts=[contrastive_answer_text],
                                                    spacy_tokenizer=spacy_tokenizer)
            # Add project(select) program-supervision for the contrastive question
            contrastive_qa_dict = add_project_select_program_supervision(contrastive_qa_dict)

            # constituency_parser_output = conparse_predictor.predict(contrastive_question)
            # contrastive_question_tokens = constituency_parser_output['tokens']
            # contrastive_questions_spans = constituency_parser_output['spans']
            # contrastive_program_node = get_contrastive_question_program(contrastive_question_tokens,
            #                                                             contrastive_questions_spans)
            # if contrastive_program_node is None:
            #     continue
            orig_program_node = node_from_dict(progsupervision)
            orig_program_lisp = nested_expression_to_lisp(orig_program_node.get_nested_expression())

            extra_annotation = {
                "orig_program_lisp": orig_program_lisp,
                "orig_question": question,
                "origprog_postorder_node_idx": 0,
                "sharedprog_postorder_node_idx": 0,   # select operation for both programs are same
            }

            contrastive_qa_dict.update(extra_annotation)

            # contrastive_q_annodict = get_substructure_annotation_dict(
            #     aux_question=contrastive_question,
            #     aux_question_tokens=contrastive_question_tokens,
            #     aux_program_node=contrastive_program_node,
            #     orig_program_lisp=orig_program_lisp,
            #     orig_question=question,
            #     origprog_postorder_node_idx=0,
            #     sharedprog_postorder_node_idx=0)  # both selects should be same

            # Adding contrastive ques annotations as DROP-style qa-dict w/ few extra keys, ones in extra_annotation
            qa[constants.shared_substructure_annotations] = [contrastive_qa_dict]
            q_w_pairedq += 1

        total_paras += 1
        # new_sample_dataset[passage_id] = passage_info
        # print(total_paras)
        # if total_paras == 10:
        #     break

    print("Num-questions: {}  Num-q w/ contrastive-questions: {}".format(total_q, q_w_pairedq))
    return squad_dataset

def get_squad_json(train_or_dev):
    squad_json = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_drop.json"
    return squad_json


def main(args):
    train_or_dev = "train"
    squad_train_json = get_squad_json(train_or_dev)
    squad_train_dataset = read_drop_dataset(squad_train_json)

    qgen_model_targz = "/shared/nitishg/checkpoints/squad-qgen/BS_6/BEAM_1/MASKQ_false/S_42/model.tar.gz"

    print("Preparing datset with contrastive questions")
    squad_dataset_w_contrastive_questions = get_contrastive_questions(squad_train_dataset, qgen_model_targz)

    output_json = f"/shared/nitishg/data/squad/squad-{train_or_dev}-v1.1_drop-wcontrastive_dep.json"
    print(f"Writing squad drop-formatted data to : {output_json}")
    with open(output_json, 'w') as outf:
        json.dump(squad_dataset_w_contrastive_questions, outf, indent=4)


if __name__=="__main__":
    main(None)