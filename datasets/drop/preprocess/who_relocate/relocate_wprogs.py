from typing import List, Dict, Tuple
import json
import os
from collections import defaultdict
import datasets.drop.constants as constants
import argparse

WHO_RELOCATE_NGRAMS = ["which player scored", "who kicked the", "who threw the", "who scored the", "who caught the"]


def readDataset(input_json):
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    return dataset


def relocate_program_qattn(tokenized_queslower: str):
    """ Here we'll annotate questions with one/two attentions depending on if the program type is
        1. find(QuestionAttention)
        2. filter(QuestionAttention, find(QuestionAttention))
    """
    question_lower = tokenized_queslower
    question_tokens: List[str] = question_lower.split(' ')
    qlen = len(question_tokens)

    qtype = None
    reloc_qattn = [0.0] * qlen
    filter_qattn = [0.0] * qlen
    find_qattn = [0.0] * qlen

    tokens_with_find_attention = ["touchdown", "run", "pass", "field", "goal", "passing", "TD", "td", "rushing",
                                  "kick", "scoring", "drive", "touchdowns", "reception", "interception", "return",
                                  "goals", "passes"]
    tokens_with_no_attention = ["which", "Which", "who", "Who", "How", "many", "yards", "yard", "was", "the",
                                "longest", "shortest", "?", "kicked", "caught", "threw", "player", "scored",
                                "of", "in", "game", "most"]

    if any(span in question_lower for span in ["who threw the", "who caught the", "who kicked the", "who scored the",
                                               "which player scored"]):
        # first deal with non longest / shortest -- strategy is everything that is not in find or relocate is filter
        reloc_qattn[1] = 1.0
        for i, t in enumerate(question_tokens):
            if t in tokens_with_find_attention:
                find_qattn[i] = 1.0
            elif t in tokens_with_no_attention:
                continue
            else:
                filter_qattn[i] = 1.0

        if "longest" in question_tokens:
            if sum(filter_qattn) != 0:
                qtype = constants.RELOC_maxfilterfind_qtype
            else:
                qtype = constants.RELOC_maxfind_qtype
        elif "shortest" in question_tokens:
            if sum(filter_qattn) != 0:
                qtype = constants.RELOC_minfilterfind_qtype
            else:
                qtype = constants.RELOC_minfind_qtype
        else:
            if sum(filter_qattn) != 0:
                qtype = constants.RELOC_filterfind_qtype
            else:
                qtype = constants.RELOC_find_qtype

    if sum(reloc_qattn) == 0:
        reloc_qattn = None
    if sum(filter_qattn) == 0:
        filter_qattn = None
    if sum(find_qattn) == 0:
        find_qattn = None

    return qtype, reloc_qattn, filter_qattn, find_qattn


def convert_span_to_attention(qlen, span):
    # span is inclusive on both ends
    qattn = [0.0] * qlen
    for x in range(span[0], span[1] + 1):
        qattn[x] = 1.0

    return qattn


def preprocess_Relocate_ques_wattn(dataset):
    """ This function prunes for questions that are count based questions.

        Along with pruning, we also supervise the with the qtype and program_supervised flag
    """

    new_dataset = {}
    total_ques = 0
    after_pruning_ques = 0
    questions_w_attn = 0
    num_passages = len(dataset)
    qtype_dist = defaultdict(int)

    for passage_id, passage_info in dataset.items():
        new_qa_pairs = []
        for question_answer in passage_info[constants.qa_pairs]:
            total_ques += 1

            original_question = question_answer[constants.cleaned_question]
            question_lower = original_question.lower()
            tokenized_ques = question_answer[constants.tokenized_question]
            tokens = tokenized_ques.split(' ')
            qlen = len(tokens)
            if any(span in question_lower for span in WHO_RELOCATE_NGRAMS):
                (qtype,
                 reloc_qattn,
                 filter_qattn,
                 find_qattn) = relocate_program_qattn(tokenized_queslower=tokenized_ques.lower())

                # if question_attention_filter_span is not None:
                #     filter_qattn = convert_span_to_attention(qlen, question_attention_filter_span)
                # else:
                #     filter_qattn = None

                # find_qattn = convert_span_to_attention(qlen, question_attention_find_span)

                if qtype is not None and reloc_qattn is not None and find_qattn is not None:
                    question_answer[constants.qtype] = qtype
                    question_answer[constants.program_supervised] = True
                    qtype_dist[qtype] += 1

                    # Inserting qattns so that the order is (reloc_attn, filter_qattn, find_qattn)
                    # Definitely a reloc attn and find.
                    qattn_tuple = []
                    qattn_tuple.append(reloc_qattn)
                    question_answer[constants.qattn_supervised] = True

                    if filter_qattn is not None:
                        qattn_tuple.append(filter_qattn)

                    if find_qattn is not None:
                        qattn_tuple.append(find_qattn)

                    # # Also adding qattn -- everything apart from the first two tokens
                    # if filter_qattn is not None:
                    #     question_answer[constants.ques_attention_supervision] = [filter_qattn, find_qattn]
                    # else:
                    #     question_answer[constants.ques_attention_supervision] = [find_qattn]
                    question_answer[constants.ques_attention_supervision] = qattn_tuple
                    question_answer[constants.qattn_supervised] = True
                    questions_w_attn += 1

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"Passages after-pruning:{num_passages_after_prune}  Question after-pruning:{after_pruning_ques}")
    print(f"Ques with attn: {questions_w_attn}")
    print(f"QType distribution: {qtype_dist}")

    return new_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    train_json = 'drop_dataset_train.json'
    dev_json = 'drop_dataset_dev.json'

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput dir: {output_dir}")

    input_trnfp = os.path.join(input_dir, train_json)
    input_devfp = os.path.join(input_dir, dev_json)
    output_trnfp = os.path.join(output_dir, train_json)
    output_devfp = os.path.join(output_dir, dev_json)

    train_dataset = readDataset(input_trnfp)
    dev_dataset = readDataset(input_devfp)

    new_train_dataset = preprocess_Relocate_ques_wattn(train_dataset)

    new_dev_dataset = preprocess_Relocate_ques_wattn(dev_dataset)

    with open(output_trnfp, 'w') as f:
        json.dump(new_train_dataset, f, indent=4)

    with open(output_devfp, 'w') as f:
        json.dump(new_dev_dataset, f, indent=4)

    print("Written count dataset")

