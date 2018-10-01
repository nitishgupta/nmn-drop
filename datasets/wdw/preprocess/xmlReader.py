import os
import sys
import json
import argparse
import xmltodict
from datasets.wdw.preprocess.wdwdoc import WDWDoc


def dumpDocsAsJson(xml_fp: str, output_path: str):
    print("Reading XML: {}".format(xml_fp))
    print("Output filepath: {}".format(output_path))
    assert(os.path.exists(xml_fp))

    obj = xmltodict.parse(open(xml_fp, 'rb'))

    outf = open(output_path, 'w')
    numdocswritten = 0

    for sample in obj['ROOT']['mc']:
        questionDict = sample['question']
        contextArtDict = sample['contextart']
        choiceDicts = sample['choice']

        qid = questionDict['@id']
        qleftContext = questionDict['leftcontext']
        qrightContext = questionDict['rightcontext']
        if qleftContext is None:
            qleftContext = ""
        if qrightContext is None:
            qrightContext = ""

        contextId = contextArtDict['@id']
        contextPara = contextArtDict['#text']

        candidateChoices = []
        correctChoice = ""
        for choicedict in choiceDicts:
            if choicedict['@correct'] == 'true':
                correctChoice = choicedict['#text']
            else:
                candidateChoices.append(choicedict['#text'])



        doc = WDWDoc(qid=qid, qleftContext=qleftContext,
                     qrightContext=qrightContext, contextId=contextId,
                     contextPara=contextPara,
                     candidateChoices=candidateChoices,
                     correctChoice=correctChoice)

        outf.write(doc.tojson())
        outf.write("\n")
        numdocswritten += 1

    outf.close()
    print("Number of docs written : {}".format(numdocswritten))



def main(args):

    wdw_strict_dir = os.path.join(args.wdw_dir, "Strict")
    wdw_relax_dir = os.path.join(args.wdw_dir, "Relaxed")

    trainfile = os.path.join(wdw_strict_dir, 'train.xml')
    relaxed_trainfile = os.path.join(wdw_relax_dir, 'train.xml')
    devfile = os.path.join(wdw_strict_dir, 'val.xml')
    testfile = os.path.join(wdw_strict_dir, 'test.xml')

    trainout = os.path.join(args.outdir, 'train.jsonl')
    relaxed_trainout = os.path.join(args.outdir, 'train_relax.jsonl')
    devout = os.path.join(args.outdir, 'val.jsonl')
    testout = os.path.join(args.outdir, 'test.jsonl')

    dumpDocsAsJson(xml_fp=trainfile, output_path=trainout)
    dumpDocsAsJson(xml_fp=relaxed_trainfile, output_path=relaxed_trainout)
    dumpDocsAsJson(xml_fp=devfile, output_path=devout)
    dumpDocsAsJson(xml_fp=testfile, output_path=testout)


if __name__ == '__main__':
    wdw_dir = '/srv/local/data/nitishg/WDW/who_did_what'
    outdir = '/srv/local/data/nitishg/WDW/raw_data'

    parser = argparse.ArgumentParser()
    parser.add_argument('--wdw_dir', default=wdw_dir)
    parser.add_argument('--outdir', default=outdir)
    args = parser.parse_args()

    main(args)