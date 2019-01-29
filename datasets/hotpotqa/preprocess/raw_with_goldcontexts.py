import copy
import time
import json
import argparse

from datasets.hotpotqa.utils import constants

import multiprocessing



def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""

    chunk_size = n
    return [iterable[i:i + chunk_size] for i in range(0, len(iterable), chunk_size)]


def onlyGoldContexts(jsonobj):
    """ Takes a hotpotqa jsonobj instance, and returns a jsonobj which only contains gold_contexts """

    new_doc = copy.deepcopy(jsonobj)

    # List with two elements (title, sent_id)
    supporting_facts = jsonobj[constants.suppfacts_field]
    set_supporting_titles = set([t for (t, sidx) in supporting_facts])
    # List of paragraphs, each represented as a tuple (title, sentences)
    contexts = jsonobj[constants.context_field]

    supporting_contexts = []
    for (title, sentences) in contexts:
        if title in set_supporting_titles:
            supporting_contexts.append((title, sentences))

    new_doc[constants.context_field] = supporting_contexts

    return new_doc


def rawOnlyGoldContexts(input_json: str, output_json: str, numproc: float) -> None:
    """ Tokenize the question, answer and context in the HotPotQA Json.

    Returns:
    --------
    Jsonl file with same datatypes as input with the modification/addition of:
    Modifications:
        q_field: The question is tokenized
        context_field: Context sentences are now tokenized, but stored with white-space delimition

    Additions:
        ans_tokenized_field: tokenized answer if needed
        q_ner_field: NER tags for question. Each NER tag is (spantext, start, end, label) with exclusive-end.
        ans_ner_field: NER tags in answer
        context_ner_field: NER tags in each of the context sentences
    """

    print("Reading input jsonl: {}".format(input_json))
    print("Output filepath: {}".format(output_json))

    # Input file contains single json obj with list of questions as jsonobjs inside it
    with open(input_json, 'r') as f:
        jsonobjs = json.load(f)

    print("Number of docs: {}".format(len(jsonobjs)))

    process_pool = multiprocessing.Pool(numproc)

    # # Makeing tuples of jsobobj with propn bool to pass as arguments
    # jsonobjs_propnbool = [jsonobj for jsonobj in jsonobjs]

    print("Making jsonobj chunks")
    jsonobj_chunks = grouper(100, jsonobjs)
    print(f"Number of chunks made: {len(jsonobj_chunks)}")

    output_jsonobjs = []
    group_num = 1

    stime = time.time()
    for chunk in jsonobj_chunks:
        # The main function that processes the input jsonobj
        result = process_pool.map(onlyGoldContexts, chunk)
        output_jsonobjs.extend(result)

        ttime = time.time() - stime
        ttime = float(ttime) / 60.0
        print(f"Groups done: {group_num} in {ttime} mins")
        group_num += 1

    print(f"Multiprocessing finished. Total elems in output: {len(output_jsonobjs)}")

    with open(output_json, 'w') as outf:
        json.dump(output_jsonobjs, outf)

    # with open(output_jsonl, 'w') as outf:
    #     for jsonobj in output_jsonobjs:
    #
    #         outf.write(json.dumps(jsonobj))
    #         outf.write("\n")
    #         numdocswritten += 1
    #         if numdocswritten % 10000 == 0:
    #             ttime = time.time() - stime
    #             ttime = float(ttime)/60.0
    #             print(f"Number of docs written: {numdocswritten} in {ttime} mins")

    print("Docs written!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_json', default=True)
    parser.add_argument('--nump', type=int, default=10)
    args = parser.parse_args()

    # args.input_json --- is the raw json from the HotpotQA dataset
    rawOnlyGoldContexts(input_json=args.input_json, output_json=args.output_json, numproc=args.nump)