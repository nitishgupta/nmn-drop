import os
import sys
import json
import argparse
from utils import util
from typing import List, Tuple, Dict


def _writeSortedDocs(input_jsonl: str, output_jsonl: str, sort_key: str, reverse: bool) -> None:
    """ Converts WDW jsonl files from xmlReader to tokenized, NP chunked jsonl files.

    Json keys:
    qid, contextId -- strings
    qleftContext, qrightContext -- list of tokens (Each considered as a single sentence)
    contextPara -- list of list of tokens
    correctChoice -- list of tokens
    candidateChoices -- list of list of tokens
    qleftContext_NPs, qrightContext_NPs -- list of int-tuples [start, end (exclusive)]
    contextPara_NPs -- list of list of int-tuples. Len of outer list == numSentences
    """

    assert os.path.exists(input_jsonl)

    print("Reading input ... ", end="", flush=True)

    with open(input_jsonl, "r") as f:

        lines = f.readlines()
        docDicts = [json.loads(s) for s in lines]

    doc_lengths = [sum([len(s) for s in doc[sort_key]]) for doc in docDicts]

    print("Sorting input ... ", end="", flush=True)

    sorted_docDicts = [x for _, x in sorted(zip(doc_lengths, docDicts), key=lambda pair: pair[0], reverse=reverse)]

    numdocswritten = 0

    print("Writing output ... ", end="", flush=True)

    with open(output_jsonl, "w") as outf:
        for doc in sorted_docDicts:
            outf.write(json.dumps(doc))
            outf.write("\n")
            numdocswritten += 1

    print("Output written!", flush=True)


def _getdocweights(docs: List[Dict], sort_keys: List[str], depths: List[int], combination: str = "x*y") -> List[int]:
    doc_weights = []
    for doc in docs:
        docw = 1 if combination == "x*y" else 0
        for key, depth in zip(sort_keys, depths):
            val = doc[key]
            len = util.count_list(input_list=val, depth=depth)
            if combination == "x*y":
                docw *= len
            elif combination == "x+y":
                docw += len
        doc_weights.append(docw)

    return doc_weights


def writeSortedDocs(
    input_jsonl: str,
    output_jsonl: str,
    sort_keys: List[str],
    depths: List[int],
    combination: str = "x*y",
    reverse: bool = True,
) -> None:
    """
    Sort input jsonl documents based on values from multiple keys.
    Each sorting criteria is a nested list, whose depth is specified. For example, counting number of mentions in 
    a list of clusters, where each cluster is a list of mentions   
    :param input_jsonl: input jsonl file
    :param output_jsonl: output jsonl file
    :param sort_keys: List of keys whose values will be used for sorting
    :param depths: the depth of the value lists corresponding to the keys 
    :param combination: combination to arrive at the final sorting order. Possible values "x*y" or "x+y"
    :param reverse: True indicates decreasing order
    :return: 
    """ ""
    possible_combinations = ["x*y", "x+y"]
    assert combination in possible_combinations, "Combination value is invalid"
    assert os.path.exists(input_jsonl), "Input jsonl doesn't exist"

    print("Reading input ... ", end="", flush=True)

    with open(input_jsonl, "r") as f:

        lines = f.readlines()
        docDicts = [json.loads(s) for s in lines]

    doc_weights = _getdocweights(docDicts, sort_keys, depths, combination)

    print("Sorting input ... ", end="", flush=True)

    sorted_docDicts = [x for _, x in sorted(zip(doc_weights, docDicts), key=lambda pair: pair[0], reverse=reverse)]

    numdocswritten = 0

    print("Writing output ... ", end="", flush=True)

    with open(output_jsonl, "w") as outf:
        for doc in sorted_docDicts:
            outf.write(json.dumps(doc))
            outf.write("\n")
            numdocswritten += 1

    print("Output written!", flush=True)


def checkSort(input_jsonl: str, sort_keys: List[str], depths: List[int], combination: str = "x*y") -> None:
    """ Checks if a jsonl is sorted in the key of sort_key. """

    assert os.path.exists(input_jsonl)

    print("Reading input ... ", end="", flush=True)

    with open(input_jsonl, "r") as f:

        lines = f.readlines()
        docDicts = [json.loads(s) for s in lines]

    print("Checking sort ... ", flush=True)

    doc_weights = _getdocweights(docDicts, sort_keys, depths, combination)

    reverse_sorteddocs = all(doc_weights[i] >= doc_weights[i + 1] for i in range(len(doc_weights) - 1))

    if reverse_sorteddocs:
        print("Docs are sorted in decreasing order")
        return None
    else:
        sorteddocs = all(doc_weights[i] <= doc_weights[i + 1] for i in range(len(doc_weights) - 1))
        if sorteddocs:
            print("Docs are sorted in increasing order")
            return None
        else:
            print("Docs are NOT sorted")
            return None


def main(args):

    reverse_bool = True if args.decreasing == "true" else False
    sort_keys = args.sort_keys.split(",")
    depths = [int(x) for x in args.depths.split(",")]
    combination = args.combination

    assert len(sort_keys) == len(depths)

    print(f"Sorting keys and depths: {sort_keys}  {depths}")
    print(f"Combination: {combination}")

    if args.checksort:
        print("Checking sort of docs in : {}".format(args.input_jsonl))
        checkSort(input_jsonl=args.input_jsonl, sort_keys=sort_keys, depths=depths, combination=combination)

    else:
        print("Sort in decreasing: {}".format(reverse_bool))
        print("Sorting doc: {}".format(args.input_jsonl))

        temp_jsonl = None

        # If output in new file
        if args.new_file:
            assert args.output_jsonl is not None, "output_jsonl for storing sorted docs is required"
            output_jsonl = args.output_jsonl
        # If output in same file --- make a temp file that'll be deleted later
        else:
            temp_jsonl = args.input_jsonl + ".bak"
            output_jsonl = temp_jsonl

        writeSortedDocs(
            input_jsonl=args.input_jsonl,
            output_jsonl=output_jsonl,
            sort_keys=sort_keys,
            depths=depths,
            reverse=reverse_bool,
            combination=combination,
        )

        # Remove old input file, replace it with temp
        if not args.new_file:
            assert temp_jsonl is not None, "temp_jsonl cannot be None here."
            os.remove(args.input_jsonl)
            os.rename(temp_jsonl, args.input_jsonl)

        print("Done sorting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--sort_keys", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--new_file", action="store_true", default=False)
    parser.add_argument("--output_jsonl")
    parser.add_argument("--checksort", action="store_true", default=False)
    parser.add_argument("--decreasing", type=str, default="true")  # or true
    parser.add_argument("--combination", type=str, default="x*y")  # or "x+y"

    args = parser.parse_args()

    main(args)
