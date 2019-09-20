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


def which_player_scored_program_qattn(tokenized_queslower: str):
    if "which player scored" not in tokenized_queslower:
        raise NotImplementedError

    question_tokens: List[str] = tokenized_queslower.split(' ')
    # TO avoid writing more rules, replacing td with touchdown
    question_tokens = ["touchdown" if x == "td" else x for x in question_tokens]
    question_lower = " ".join(question_tokens)

    # Reduce number of surface forms for 'semantically-equivalent' questions
    question_lower = question_lower.replace("in the game", "of the game")
    # Replacing two token events w/ same event
    question_lower = question_lower.replace("touchdown pass", "touchdown run")
    question_lower = question_lower.replace("rushing touchdown", "touchdown run")
    question_lower = question_lower.replace("touchdown reception", "touchdown run")

    qlen = len(question_tokens)

    qtype = None
    reloc_qattn = [0.0] * qlen
    filter_qattn = [0.0] * qlen
    find_qattn = [0.0] * qlen

    reloc_qattn[0:3] = [1.0, 1.0, 1.0]  # which player scored

    if (question_lower in ["which player scored the last touchdown of the game ?",
                           "which player scored the first touchdown of the game ?",
                           "which player scored the final touchdown of the game ?",
                           "which player scored the first touchdown ?",
                           "which player scored the last touchdown ?",
                           "which player scored the final touchdown ?",
                           "which player scored the first points of the game ?",
                           "which player scored the last points of the game ?",
                           "which player scored the final points of the game ?"
                           ]):
        qtype = constants.RELOC_find_qtype
        find_qattn[4:6] = [1.0, 1.0]    # last touchdown

    if (question_lower in ["which player scored the last field goal of the game ?",
                           "which player scored the first field goal of the game ?",
                           "which player scored the final field goal of the game ?",
                           "which player scored the first field goal ?",
                           "which player scored the last field goal ?",
                           "which player scored the final field goal ?",
                           "which player scored the last touchdown run ?",
                           "which player scored the final touchdown run ?",
                           "which player scored the first touchdown run ?",
                           "which player scored the last touchdown run of the game ?",
                           "which player scored the final touchdown run of the game ?",
                           "which player scored the first touchdown run of the game ?"
                           ]):
        qtype = constants.RELOC_find_qtype
        find_qattn[4:7] = [1.0, 1.0, 1.0]  # last touchdown

    if (question_lower in ["which player scored the longest field goal of the game ?",
                           "which player scored the shortest field goal of the game ?",
                           "which player scored the longest field goal ?",
                           "which player scored the shortest field goal ?",
                           "which player scored the longest touchdown run ?",
                           "which player scored the shortest touchdown run ?",
                           "which player scored the longest touchdown run of the game ?",
                           "which player scored the shortest touchdown run of the game ?"]):
        if "longest" in question_lower:
            qtype = constants.RELOC_maxfind_qtype
        elif "shortest" in question_lower:
            qtype = constants.RELOC_minfind_qtype
        else:
            raise NotImplementedError
        find_qattn[5:7] = [1.0, 1.0]        # field goal / touchdown reception / rushing touchdown

    if (question_lower in ["which player scored the longest touchdown of the game ?",
                           "which player scored the shortest touchdown of the game ?",
                           "which player scored the longest touchdown ?",
                           "which player scored the shortest touchdown ?"]):
        if "longest" in question_lower:
            qtype = constants.RELOC_maxfind_qtype
        elif "shortest" in question_lower:
            qtype = constants.RELOC_minfind_qtype
        else:
            raise NotImplementedError
        find_qattn[5:6] = [1.0]             # touchdown

    if sum(reloc_qattn) == 0:
        reloc_qattn = None
    if sum(filter_qattn) == 0:
        filter_qattn = None
    if sum(find_qattn) == 0:
        find_qattn = None

    return qtype, find_qattn, filter_qattn, reloc_qattn


def who_X_the_program_qattn(tokenized_queslower: str):
    if not any(span in tokenized_queslower for span in ["who threw the", "who caught the",
                                                        "who kicked the", "who scored the"]):
        print(tokenized_queslower)
        raise NotImplementedError

    question_tokens: List[str] = tokenized_queslower.split(' ')
    # TO avoid writing more rules, replacing td with touchdown
    question_tokens = ["touchdown" if x == "td" else x for x in question_tokens]
    question_tokens = ["VERB" if (x == "threw" or x == "kicked" or x == "caught" or x == "scored")
                       else x for x in question_tokens]
    question_lower = " ".join(question_tokens)

    # Reduce number of surface forms for 'semantically-equivalent' questions
    question_lower = question_lower.replace("in the game", "of the game")
    # Replacing two token events w/ same event
    question_lower = question_lower.replace("touchdown pass", "touchdown run")
    question_lower = question_lower.replace("rushing touchdown", "touchdown run")


    qlen = len(question_tokens)

    qtype = None
    reloc_qattn = [0.0] * qlen
    filter_qattn = [0.0] * qlen
    find_qattn = [0.0] * qlen

    reloc_qattn[0:2] = [1.0, 1.0]  # who VERB

    #   RELOCATE ( FIND )
    if (question_lower in ["who VERB the first touchdown ?",
                           "who VERB the last touchdown ?",
                           "who VERB the final touchdown ?",
                           "who VERB the first touchdown of the game ?",
                           "who VERB the last touchdown of the game ?",
                           "who VERB the final touchdown of the game ?",
                           "who VERB the first points ?",
                           "who VERB the last points ?",
                           "who VERB the final points ?",
                           "who VERB the first points of the game ?",
                           "who VERB the last points of the game ?",
                           "who VERB the final points of the game ?"]):
        qtype = constants.RELOC_find_qtype
        find_qattn[3:5] = [1.0, 1.0]  # first touchdown

    if (question_lower in ["who VERB the first field goal ?",
                           "who VERB the last field goal ?",
                           "who VERB the final field goal ?",
                           "who VERB the first field goal of the game ?",
                           "who VERB the last field goal of the game ?",
                           "who VERB the final field goal of the game ?",
                           "who VERB the first touchdown run ?",
                           "who VERB the last touchdown run ?",
                           "who VERB the final touchdown run ?",
                           "who VERB the first touchdown run of the game ?",
                           "who VERB the last touchdown run of the game ?",
                           "who VERB the final touchdown run of the game ?"
                           ]):
        qtype = constants.RELOC_find_qtype
        find_qattn[3:6] = [1.0, 1.0, 1.0]  # first field goal

    #   RELOCATE ( MAX/MIN ( FIND ) )
    if (question_lower in ["who VERB the longest touchdown ?",
                           "who VERB the shortest touchdown ?",
                           "who VERB the longest touchdown of the game ?",
                           "who VERB the shortest touchdown of the game ?"]):
        if "longest" in question_lower:
            qtype = constants.RELOC_maxfind_qtype
        elif "shortest" in question_lower:
            qtype = constants.RELOC_minfind_qtype
        find_qattn[4:5] = [1.0]  # touchdown

    if (question_lower in ["who VERB the longest field goal ?",
                           "who VERB the shortest field goal ?",
                           "who VERB the longest field goal of the game ?",
                           "who VERB the shortest field goal of the game ?",
                           "who VERB the longest touchdown run ?",
                           "who VERB the shortest touchdown run ?",
                           "who VERB the longest touchdown run of the game ?",
                           "who VERB the shortest touchdown run of the game ?"]):
        if "longest" in question_lower:
            qtype = constants.RELOC_maxfind_qtype
        elif "shortest" in question_lower:
            qtype = constants.RELOC_minfind_qtype
        find_qattn[4:6] = [1.0, 1.0]  # field goal

    #   RELOCATE ( MAX/MIN ( FILTER ( FIND ) ) )
    if (question_lower in ["who VERB the longest touchdown of the first half ?",
                           "who VERB the shortest touchdown of the first half ?",
                           "who VERB the longest touchdown of the second half ?",
                           "who VERB the shortest touchdown of the second half ?",
                           "who VERB the longest touchdown of the first quarter ?",
                           "who VERB the shortest touchdown of the first quarter ?"
                           "who VERB the longest touchdown of the second quarter ?",
                           "who VERB the shortest touchdown of the second quarter ?",
                           "who VERB the longest touchdown of the third quarter ?",
                           "who VERB the shortest touchdown of the third quarter ?",
                           "who VERB the longest touchdown of the fourth quarter ?",
                           "who VERB the shortest touchdown of the fourth quarter ?",
                           ]):
        if "longest" in question_lower:
            qtype = constants.RELOC_maxfilterfind_qtype
        elif "shortest" in question_lower:
            qtype = constants.RELOC_minfilterfind_qtype
        else:
            raise NotImplementedError
        find_qattn[4:5] = [1.0]             # touchdown
        filter_qattn[7:9] = [1.0, 1.0]      # first half

    if (question_lower in ["who VERB the longest field goal of the first half ?",
                           "who VERB the shortest field goal of the first half ?",
                           "who VERB the longest field goal of the second half ?",
                           "who VERB the shortest field goal of the second half ?",
                           "who VERB the longest field goal of the first quarter ?",
                           "who VERB the shortest field goal of the first quarter ?"
                           "who VERB the longest field goal of the second quarter ?",
                           "who VERB the shortest field goal of the second quarter ?",
                           "who VERB the longest field goal of the third quarter ?",
                           "who VERB the shortest field goal of the third quarter ?",
                           "who VERB the longest field goal of the fourth quarter ?",
                           "who VERB the shortest field goal of the fourth quarter ?",
                           "who VERB the longest touchdown run of the first half ?",
                           "who VERB the shortest touchdown run of the first half ?",
                           "who VERB the longest touchdown run of the second half ?",
                           "who VERB the shortest touchdown run of the second half ?",
                           "who VERB the longest touchdown run of the first quarter ?",
                           "who VERB the shortest touchdown run of the first quarter ?"
                           "who VERB the longest touchdown run of the second quarter ?",
                           "who VERB the shortest touchdown run of the second quarter ?",
                           "who VERB the longest touchdown run of the third quarter ?",
                           "who VERB the shortest touchdown run of the third quarter ?",
                           "who VERB the longest touchdown run of the fourth quarter ?",
                           "who VERB the shortest touchdown run of the fourth quarter ?",
                           ]):
        if "longest" in question_lower:
            qtype = constants.RELOC_maxfilterfind_qtype
        elif "shortest" in question_lower:
            qtype = constants.RELOC_minfilterfind_qtype
        else:
            raise NotImplementedError
        find_qattn[4:6] = [1.0, 1.0]        # field goal
        filter_qattn[8:10] = [1.0, 1.0]     # first half

    if sum(reloc_qattn) == 0:
        reloc_qattn = None
    if sum(filter_qattn) == 0:
        filter_qattn = None
    if sum(find_qattn) == 0:
        find_qattn = None

    return qtype, find_qattn, filter_qattn, reloc_qattn





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

    # if any(span in question_lower for span in ["who threw the", "who caught the", "who kicked the"]):
    #     # Non-filter question
    #     if question_lower in ["who threw the longest touchdown ?", "who threw the longest td pass ?",
    #                           "who threw the longest touchdown pass of the game ?",
    #                           "who threw the longest touchdown pass ?",
    #                           "who kicked the longest field goal ?", "who kicked the longest field goal of the game?",
    #                           "who caught the longest touchdown pass ?",
    #                           "who caught the longest touchdown pass of the game ?",
    #                           "who caught the longest pass ?", "who caught the longest pass of the game ?",
    #                           "who caught the longest td pass?", "who caught the longest td pass of the game ?",
    #                           ]:
    #         qtype = constants.RELOC_maxfind_qtype
    #         reloc_qattn[1] = 1.0
    #         for i, t in enumerate(question_lower.split(' ')):
    #             if t in tokens_with_find_attention:
    #                 find_qattn[i] = 1.0
    #
    #     elif question_lower in ["who threw the shortest touchdown ?", "who threw the shortest td pass ?",
    #                             "who threw the shortest touchdown pass of the game ?",
    #                             "who threw the shortest touchdown pass ?",
    #                             "who kicked the shortest field goal ?",
    #                             "who kicked the shortest field goal of the game ?",
    #                             "who caught the shortest touchdown pass ?",
    #                             "who caught the shortest touchdown pass of the game ?",
    #                             "who caught the shortest pass ?", "who caught the shortest pass of the game ?",
    #                             "who caught the shortest td pass?", "who caught the shortest td pass of the game ?",
    #                             ]:
    #         qtype = constants.RELOC_minfind_qtype
    #         reloc_qattn[1] = 1.0
    #         for i, t in enumerate(question_lower.split(' ')):
    #             if t in tokens_with_find_attention:
    #                 find_qattn[i] = 1.0
    #
    #     else:
    #         pass
    if any(span in question_lower for span in ["who threw the", "who caught the", "who kicked the", "who scored the",
                                               "which player scored"]):
    #elif any(span in question_lower for span in ):
        # These questions are relocate(find), relocate(filter(find)),
        # relocate(maxnum(find)), or relocate(maxnum(filter(find))))

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


def make_supervision_dict(dataset):
    basic_keys = [constants.program_supervised, constants.qattn_supervised, constants.exection_supervised]
    qtype_dict = defaultdict(int)
    total_num_qa = 0
    supervision_dict = defaultdict(int)
    for passage_idx, passage_info in dataset.items():
        total_num_qa += len(passage_info[constants.qa_pairs])
        for qa in passage_info[constants.qa_pairs]:
            if constants.qtype in qa:
                qtype_dict[qa[constants.qtype]] += 1

            all_basic_true = False
            for key in basic_keys:
                if key in qa:
                    supervision_dict[key] += 1 if qa[key] else 0
                    all_basic_true = True if qa[key] else False
                else:
                    all_basic_true = False
            if all_basic_true:
                supervision_dict[constants.strongly_supervised] += 1

    return supervision_dict, qtype_dict



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
                if "which player scored" in question_lower:
                    (qtype,
                     reloc_qattn,
                     filter_qattn,
                     find_qattn) = which_player_scored_program_qattn(tokenized_ques.lower())
                else:
                    (qtype,
                     reloc_qattn,
                     filter_qattn,
                     find_qattn) = who_X_the_program_qattn(tokenized_ques.lower())

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

                    question_answer[constants.ques_attention_supervision] = qattn_tuple
                    question_answer[constants.qattn_supervised] = True
                    questions_w_attn += 1

                    print()

                new_qa_pairs.append(question_answer)

        if len(new_qa_pairs) > 0:
            passage_info[constants.qa_pairs] = new_qa_pairs
            new_dataset[passage_id] = passage_info
            after_pruning_ques += len(new_qa_pairs)

    supervision_dict, _ = make_supervision_dict(new_dataset)

    num_passages_after_prune = len(new_dataset)
    print(f"Passages original:{num_passages}  Questions original:{total_ques}")
    print(f"Passages after-pruning:{num_passages_after_prune}  Question after-pruning:{after_pruning_ques}")
    print(f"Supervision dict: {supervision_dict}")
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

