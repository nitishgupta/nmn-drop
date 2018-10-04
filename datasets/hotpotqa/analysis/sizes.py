import os
import json
import argparse
from typing import List, Tuple, Any
from datasets.hotpotqa.utils import constants
from utils import util

ner_type: List[Any]

def statsOnListOfList(inplist: List[List[Any]], debug=False):
    """ Stats for a list of list of strs.
    For example, a list of sentence-split documents, document with tokenized sentences, or a list of NERs per sent.

    Returns:
    --------
    numexamples: Total size of the outer list, number of examples i.e.
    total_numelems: Cumulative size of the inner list
    avg_numelems: Avg size of inner list, i.e. total_numelems / numexamples
    min_numelems: Min len of the inner list
    max_numelems: Max len of the outer list
    """

    numexamples = len(inplist)

    total_numelems = 0
    max_numelems = float('-inf')
    min_numelems = float('inf')

    biggest_elem = None

    for example in inplist:
        # List of elems
        example: List[Any] = example
        total_numelems += len(example)
        if len(example) > max_numelems:
            biggest_elem = example

        max_numelems = len(example) if len(example) > max_numelems else max_numelems
        min_numelems = len(example) if len(example) < min_numelems else min_numelems

    avg_numelems = float(total_numelems)/numexamples

    if debug:
        print('\n'.join(biggest_elem))

    return (numexamples, total_numelems, avg_numelems, min_numelems, max_numelems)


def printListofListStats(example: str, elem: str, stats):
    (numexamples, total_numelems, avg_numelems, min_numelems, max_numelems) = stats
    avg_numelems = util.round_all(avg_numelems, 3)
    print(f"Total {example}: {numexamples}")
    print(f"Total {elem} across {example}: {total_numelems}")
    print(f"Avg {elem} per {example}: {avg_numelems}")
    print(f"Min {elem} in any {example}: {min_numelems}")
    print(f"Max {elem} in any {example}: {max_numelems}")


def dataStats(input_jsonl: str) -> None:
    print("Reading dataset: {}".format(input_jsonl))
    qa_examples = util.readJsonlDocs(input_jsonl)

    # List of contexts. Each context is a list of sentences.
    all_context_paras: List[List[str]] = []

    # List of sentences_perqa. Each sentences_perqa is a list of sentences in all contexts of the Q.
    sentences_perqa: List[List[str]] = []

    # List of context_ners for each context. Each context_ner is a list of ner tags.
    all_context_ners: List[List[Any]] = []

    # List of contexts for each q. Each context is a str. Size: number of questions
    contexts_perqa: List[List[str]] = []

    # List of tokenized_context for each context para. Each tokenized_context is a list of tokens.
    all_qa_tokenizedcontexts: List[List[str]] = []
    # List of ners for each q.
    all_qa_ners: List[List[str]] = []

    # List of sentences. Each sentence is a list of tokens. Size: number of sentences in dataset
    all_tokenized_sents: List[List[str]] = []

    questions: List[List[str]] = []
    ques_ners: List[List[Any]] = []

    answers: List[List[str]] = []
    ans_ners: List[List[Any]] = []

    for qaexample in qa_examples:
        # List of contexts. Each context is a list of (Title, [sentence])
        contexts: List[Tuple[str, List[str]]] = qaexample[constants.context_field]
        # For each context, for each sentence, a list of ners
        ners: List[List[List[ner_type]]] = qaexample[constants.context_ner_field]

        # List of contexts, each one a list of sent_str
        context_paras: List[List[str]] = [sentences for (_, sentences) in contexts]
        all_context_paras.extend(context_paras)
        sentences: List[str] = [sent for context in context_paras for sent in context]
        sentences_perqa.append(sentences)

        # List of context ners
        q_ners = []
        for context_ners in ners:
            ners_in_context = []
            for sent_ners in context_ners:
                ners_in_context.extend(sent_ners)
                q_ners.extend(sent_ners)
            all_context_ners.append(ners_in_context)
        all_qa_ners.append(q_ners)

        # all_context_ners.extend(context_ners)

        # List of contexts (as a single str) for the question
        full_contexts: List[str] = [' '.join(sentences) for (_, sentences) in contexts]
        contexts_perqa.append(full_contexts)

        # List of contexts, each a list of tokens
        full_tokenizedcontext: List[List[str]] = [' '.join(sentences).split(' ') for (_, sentences) in contexts]
        all_qa_tokenizedcontexts.extend(full_tokenizedcontext)

        sentences: List[List[str]] = [sent.split(" ") for sentences in context_paras for sent in sentences]

        all_tokenized_sents.extend(sentences)


        ### Question processing
        questions.append(qaexample[constants.q_field].split(' '))
        ques_ners.append(qaexample[constants.q_ner_field])

        ### Answer processing
        answers.append(qaexample[constants.ans_field].split(' '))
        ans_ners.append(qaexample[constants.ans_ner_field])

    stats = statsOnListOfList(contexts_perqa)
    printListofListStats(example="questions", elem="contexts", stats=stats)
    print()

    stats = statsOnListOfList(sentences_perqa)
    printListofListStats(example="questions", elem="sentences", stats=stats)
    print()

    stats = statsOnListOfList(all_context_paras)
    printListofListStats(example="Contexts", elem="Sentences", stats=stats)
    print()

    stats = statsOnListOfList(all_qa_tokenizedcontexts)
    printListofListStats(example="Contexts", elem="Tokens", stats=stats)
    print()

    stats = statsOnListOfList(all_context_ners)
    printListofListStats(example="Contexts", elem="Ners", stats=stats)
    num_nonercontext = 0
    for (context_ner, context) in zip(all_context_ners, all_context_paras):
        if len(context_ner) == 0:
            num_nonercontext += 1
    print(f"Number of contexts with no ner: {num_nonercontext}")
    print()

    stats = statsOnListOfList(all_qa_ners)
    printListofListStats(example="Questions", elem="Ners", stats=stats)
    print()

    print("QUESTION STATS")
    stats = statsOnListOfList(questions)
    printListofListStats(example="Questions", elem="Tokens", stats=stats)
    print()
    stats = statsOnListOfList(ques_ners)
    printListofListStats(example="Questions", elem="NERs", stats=stats)
    num_nonerques = 0
    for qner, ques in zip(ques_ners, questions):
        if len(qner) == 0:
            num_nonerques += 1
            print(' '.join(ques))
    print(f"Number of ques with no ner: {num_nonerques}")
    print()


    print("Ans STATS")
    stats = statsOnListOfList(answers)
    printListofListStats(example="Answers", elem="Tokens", stats=stats)
    print()
    stats = statsOnListOfList(ans_ners)
    printListofListStats(example="Answers", elem="NERs", stats=stats)
    num_nonerans = 0
    for (ans_ner, ans) in zip(ans_ners, answers):
        if len(ans_ner) == 0:
            num_nonerans += 1
    print(f"Number of answers with no ner: {num_nonerans}")








def main(args):

    dataStats(input_jsonl=args.input_jsonl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    args = parser.parse_args()

    main(args)
