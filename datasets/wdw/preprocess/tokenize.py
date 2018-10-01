import os
import json
import argparse
from ccg_nlpy import local_pipeline
from utils import TAUtils, util
from datasets.wdw.preprocess.wdwdoc import WDWDoc

lp = local_pipeline.LocalPipeline()


def tokenizeWDWDocs(input_jsonl: str, output_jsonl: str) -> None:
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

    print("Reading input jsonl: {}".format(input_jsonl))
    print("Output filepath: {}".format(output_jsonl))

    docs = util.readJsonlDocs(input_jsonl)

    print("Number of docs: {}".format(len(docs)))
    numdocswritten = 0
    with open(output_jsonl, 'w') as outf:
        for doc in docs:
            new_doc = {}
            new_doc['qid'] = doc['qid']
            new_doc['contextId'] = doc['contextId']

            # Tokenize context, and split sentences larger than a threshold (=120)
            ta_para = lp.doc_split_on_hyphens(doc['contextPara'])
            ta_para = TAUtils.thresholdSentLength(ta=ta_para, lp=lp)
            sentences = TAUtils.get_sentences(ta_para)
            new_doc['context'] = sentences

            # POS Tagging for context
            new_doc['context_pos'] = TAUtils.getPOS_perSent(ta_para)

            # NP Chunking for context
            nps_withinSentIdxs = TAUtils.getNPChunks_perSent(ta_para)
            assert len(nps_withinSentIdxs) == len(sentences)
            new_doc['context_nps'] = nps_withinSentIdxs

            # For left context, if not empty
            if doc['qleftContext']:
                ta_left = lp.doc_split_on_hyphens(doc['qleftContext'])
                new_doc['qleft_context'] = ta_left.tokens
                nps_left = TAUtils.getNPsWithGlobalOffsets(ta_left)
                new_doc['qleft_nps'] = nps_left
                new_doc['qleft_pos'] = TAUtils.getPOS_forDoc(ta_left)
            else:
                new_doc['qleft_context'] = []
                new_doc['qleft_pos'] = []
                new_doc['qleft_nps'] = []

            # For right context, if not empty
            if doc['qrightContext']:
                ta_right = lp.doc_split_on_hyphens(doc['qrightContext'])
                new_doc['qright_context'] = ta_right.tokens
                new_doc['qright_nps'] = TAUtils.getNPsWithGlobalOffsets(ta_right)
                new_doc['qright_pos'] = TAUtils.getPOS_forDoc(ta_right)
            else:
                new_doc['qright_context'] = []
                new_doc['qright_nps'] = []
                new_doc['qright_pos'] = []

            ta_correctchoice = lp.doc_split_on_hyphens(doc['correctChoice'])
            new_doc['answer'] = ta_correctchoice.tokens

            ta_candidates = [lp.doc_split_on_hyphens(c)
                             for c in doc['candidateChoices']]

            new_doc['candidates'] = [ta.tokens for ta in ta_candidates]

            outf.write(json.dumps(new_doc))
            outf.write("\n")
            numdocswritten += 1
            if numdocswritten % 10000 == 0:
                print("Number of docs written: {}".format(numdocswritten))

    print("Number of docs written: {}".format(numdocswritten))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfpath', required=True)
    parser.add_argument('--outputfpath', default=True)
    args = parser.parse_args()

    tokenizeWDWDocs(input_jsonl=args.inputfpath, output_jsonl=args.outputfpath)