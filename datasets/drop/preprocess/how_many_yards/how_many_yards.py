from typing import List, Dict, Tuple
import json
from nltk.corpus import stopwords
import os
from collections import defaultdict
import datasets.drop.constants as constants
import argparse

""" This script is used to augment date-comparison-data by flipping events in the questions """
THRESHOLD = 20

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(["'s", ","])


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def get_question_attention(tokenized_question: str) -> List[float]:
    ques_tokens = tokenized_question.split(' ')
    attention = [0] * len(ques_tokens)

    # First 4 tokens are `How many yards was` -- no attention
    # the - if the 5th token
    # ?, longest, shortest, second,  -- no attention
    no_attend_tokens = ['second', 'longest', 'shortest', '?']
    for i, token in enumerate(ques_tokens):
        if i < 4:
            continue
        elif i == 4 and token == 'the':
            continue
        elif token in no_attend_tokens:
            continue
        else:
            attention[i] = 1

    return attention



def get_number_distribution_supervision(tokenized_question, tokenized_passage, num_answer,
                                        attention, passage_num_mens, passage_num_entidxs, passage_num_vals):
    WINDOW = 10
    passage_tokens = tokenized_passage.split(' ')
    question_tokens = tokenized_question.split(' ')

    # Only supervised longest / shortest questions
    if 'longest' not in question_tokens and 'shortest' not in question_tokens:
        return None, None
    if num_answer is None:
        return None, None

    # These are the relevant tokens in the question. We'd like to find numbers that are surrounded by these tokens
    attended_tokens = [token for att, token in zip(attention, question_tokens) if att > 0]
    attended_tokens = set(attended_tokens)
    # Replacing TD with touchdown
    if 'TD' in attended_tokens:
        attended_tokens.remove('TD')
        attended_tokens.add('touchdown')
    irrelevant_tokens = ["'", "'s", "of", "the", "game", "games"]
    for t in irrelevant_tokens:
        if t in attended_tokens:
            attended_tokens.remove(t)

    # Num of passage number tokens
    number_token_idxs = [x for (_, x, _) in passage_num_mens]

    relevant_number_tokenidxs = []
    relevant_number_entidxs = []
    relevant_number_values = []

    for menidx, number_token_idx in enumerate(number_token_idxs):
        try:
            if passage_tokens[number_token_idx + 1] != '-' or passage_tokens[number_token_idx + 2] != 'yard':
                continue
        except:
            continue
        starting_tokenidx = max(0, number_token_idx - WINDOW)   # Inclusive
        ending_tokenidx = min(len(passage_tokens), number_token_idx + WINDOW + 1)   # Exclusive
        surrounding_passage_tokens = set(passage_tokens[starting_tokenidx:ending_tokenidx])
        if 'TD' in surrounding_passage_tokens:
            surrounding_passage_tokens.remove('TD')
            surrounding_passage_tokens.add('touchdown')
        intersection_tokens = surrounding_passage_tokens.intersection(attended_tokens)
        if intersection_tokens == attended_tokens:
            relevant_number_tokenidxs.append(number_token_idx)
            relevant_number_entidxs.append(passage_num_entidxs[menidx])
            relevant_number_values.append(passage_num_vals[passage_num_entidxs[menidx]])

    if relevant_number_entidxs:
        number_grounding = [0] * len(passage_num_vals)
        number_values = set()
        for entidx in relevant_number_entidxs:
            number_grounding[entidx] = 1
            number_values.add(passage_num_vals[entidx])
        number_grounding = [number_grounding]
        number_values = [list(number_values)]
        if num_answer not in number_values[0]: # It's now a list
            number_grounding = None
            number_values = None

    else:
        number_grounding = None
        number_values = None

    # print(tokenized_question)
    # print(attended_tokens)
    # print(f"Answer: {num_answer}")
    # print(tokenized_passage)
    # print(passage_num_vals)
    # print(number_values)
    # print()

    return number_grounding, number_values



def preprocess_HowManyYardsWasThe_ques(dataset, ques_attn: bool, number_supervision: bool):
    """ This function prunes questions that start with "How many yards was".
        Mostly, longest, shortest style questions. We can also prune for these; look at longestshortest_ques.py

        Along with pruning, we also supervise the longest/shortest/second longest/second shortest questions
        by adding the question_type for those questions.

        Currently ---
        We prune out questions like `the second longest / shortest`.
        Still does prune questions like `Tom Brady's second longest/shortest` infact we label them as longest/shortest
        instead of second longest/shortest. But their size is minuscule

        Question-attention
        If the `ques_attn` flag is ON, we also add question-attention supervision

    """

    longest_question_ngram = "how many yards was the longest"
    shortest_question_ngram = "how many yards was the shortest"
    second_longest_question_ngram = "how many yards was the second longest"
    second_shortest_question_ngram = "how many yards was the second shortest"

    how_many_yards_was = "how many yards was"

    longest_qtype = constants.YARDS_longest_qtype
    shortest_qtype = constants.YARDS_shortest_qtype
    # second_longest_qtype = 'how_many_yards_second_longest'
    # second_shortest_qtype = 'how_many_yards_second_shortest'

    findnumber_qtype = constants.YARDS_findnum_qtype

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_qtypes = 0
    questions_w_attn = 0
    questions_w_numground = 0
    qtype_dist = defaultdict(int)
    num_passages = len(dataset)
    counter = 1

    for passage_id, passage_info in dataset.items():

        passage_num_mens = passage_info[constants.passage_num_mens]
        passage_num_entidxs = passage_info[constants.passage_num_entidx]
        passage_num_vals = passage_info[constants.passage_num_normalized_values]
        tokenized_passage = passage_info[constants.tokenized_passage]

        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            answer = question_answer[constants.answer]

            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            tokenized_question = question_answer[constants.tokenized_question]
            question_lower = original_question.lower()

            if how_many_yards_was in question_lower:
                if longest_question_ngram in question_lower:
                    question_answer[constants.qtype] = constants.YARDS_longest_qtype
                    question_answer[constants.program_supervised] = True
                    qtype_dist[constants.YARDS_longest_qtype] += 1
                    questions_w_qtypes += 1

                elif shortest_question_ngram in question_lower:
                    question_answer[constants.qtype] = constants.YARDS_shortest_qtype
                    question_answer[constants.program_supervised] = True
                    qtype_dist[constants.YARDS_shortest_qtype] += 1
                    questions_w_qtypes += 1

                # Skip second longest/shortest questions for now
                elif second_longest_question_ngram in question_lower:
                    continue
                elif second_shortest_question_ngram in question_lower:
                    continue

                # WHP -- a findNumber question unless longest or shortest token in the question
                else:
                    if 'longest' in question_lower:
                        question_answer[constants.qtype] = constants.YARDS_longest_qtype
                        question_answer[constants.program_supervised] = True
                        qtype_dist[longest_qtype] += 1
                        questions_w_qtypes += 1
                    elif 'shortest' in question_lower:
                        question_answer[constants.qtype] = constants.YARDS_shortest_qtype
                        question_answer[constants.program_supervised] = True
                        qtype_dist[shortest_qtype] += 1
                        questions_w_qtypes += 1
                    else:
                        question_answer[constants.qtype] = constants.YARDS_findnum_qtype
                        question_answer[constants.program_supervised] = True
                        qtype_dist[findnumber_qtype] += 1
                        questions_w_qtypes += 1

                num_answer_str = answer['number']
                num_answer = float(num_answer_str) if num_answer_str else None

                if ques_attn is True:
                    ques_attention = get_question_attention(tokenized_question)

                    if sum(ques_attention) > 0:
                        if number_supervision is True:
                            number_grounding, number_values = get_number_distribution_supervision(
                                tokenized_question, tokenized_passage, num_answer,
                                ques_attention, passage_num_mens, passage_num_entidxs,
                                passage_num_vals)
                            if number_grounding is not None:
                                question_answer[constants.qattn_supervised] = True
                                question_answer[constants.ques_attention_supervision] = [ques_attention]
                                question_answer[constants.exection_supervised] = True
                                question_answer[constants.qspan_numgrounding_supervision] = number_grounding
                                question_answer[constants.qspan_numvalue_supervision] = number_values
                                questions_w_numground += 1
                                questions_w_attn += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  After Pruning:{num_passages_after_prune}")
    print(f"Questions original:{total_ques}  After pruning:{after_pruning_ques}")
    print(f"Num of QA with qtypes and program supervised: {questions_w_qtypes}")
    print(f"Num of QA with attention supervised: {questions_w_attn}")
    print(f"Num of QA with num-grounding supervised: {questions_w_numground}")
    print(f"Qtype dist: {qtype_dist}")

    return new_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--qattn', action='store_true', default=False)
    parser.add_argument('--numground', action='store_true', default=False)
    args = parser.parse_args()

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

    input_dir = args.input_dir
    output_dir = args.output_dir
    qattn = args.qattn
    numbergrounding = args.numground

    if numbergrounding is True:
        assert qattn is True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput dir: {output_dir}")
    print(f"\nQuestion Attention annotation: {qattn}")
    print(f"\nNumber Grounding annotation: {numbergrounding}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    print()
    new_train_dataset = preprocess_HowManyYardsWasThe_ques(train_dataset, ques_attn=qattn,
                                                           number_supervision=numbergrounding)
    print()
    new_dev_dataset = preprocess_HowManyYardsWasThe_ques(dev_dataset, ques_attn=qattn,
                                                         number_supervision=numbergrounding)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written HowManyYards datasets")

''' DATASET CREATED THIS WAY

input_dir = "./resources/data/drop_old/analysis/ngram/num/how_many_yards_was_the/"
output_dir = "./resources/data/drop_old/num/how_many_yards_was_the"

'''
