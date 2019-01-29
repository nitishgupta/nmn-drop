import sys
import ujson as json
import random

import argparse
from typing import List, Tuple, Any, Dict
from datasets.hotpotqa.utils import constants
from utils import util
import datasets.hotpotqa.analysis.hotpot_evaluate_v1 as evaluation

random.seed(1)

"""
Write Question, Answer and Supporting fact paragraphs in text format for manual analysis.
"""


def _getOutputStringForQA(qa: Dict):
    '''
    Single string output for a single question/answer for human analysis of decomposition, and answer reasoning

    qa : ``Dict``
    Json dict from tokenized documents
    '''


    def _mens_str(mens):
        mens_str = ''
        for m in mens:
            mens_str += m[0] + f" ({m[1]}-{m[2]})" + ', '
        return mens_str


    question: str = qa[constants.q_field]
    answer: str = qa[constants.ans_field]
    qlevel: str = qa[constants.qlevel_field]
    qid: str = qa[constants.id_field]

    all_ne_mens = qa[constants.context_ent_ner_field]
    all_num_mens = qa[constants.context_num_ner_field]
    all_date_mens = qa[constants.context_date_ner_field]

    # List of (title, sent_id) tuples
    supporting_facts = qa[constants.suppfacts_field]
    supporting_titles = set([title for title, _ in supporting_facts])

    # List of (title, context)
    contexts = qa[constants.context_field]

    supporting_contexts_widxs = [(c_idx, context) for (c_idx, (title, context)) in enumerate(contexts) if title in supporting_titles]

    # This will later be joined with '\n' so add accordingly
    supporting_contexts_strlist = []
    for i, (c_idx, context) in enumerate(supporting_contexts_widxs):
        ne_mens, num_mens, date_mens = all_ne_mens[c_idx], all_num_mens[c_idx], all_date_mens[c_idx]
        supporting_contexts_strlist.append('---------------------------')
        supporting_contexts_strlist.append("Context #" + str(i))
        supporting_contexts_strlist.append(context)
        supporting_contexts_strlist.append(f"NE mens: {_mens_str(ne_mens)}")
        supporting_contexts_strlist.append(f"Num mens: {_mens_str(num_mens)}")
        supporting_contexts_strlist.append(f"Date mens: {_mens_str(date_mens)}")
    supporting_contexts_strlist.append('---------------------------')
    supporting_contexts_strlist.extend(['', 2*'**************************************************', ''])


    final_str = f"Q_id: {qid}" + '\n'
    final_str += f"level: {qlevel}" + '\n'
    final_str += f'question: {question}' + '\n'
    final_str += f'answer: {answer}' + '\n'
    final_str += '\n'.join(supporting_contexts_strlist)

    final_str += '\n'

    return final_str


def writeQAStrings(input_jsonl: str, output_txt: str, numq_perlevel: int=30) -> None:
    '''
    Write QA, id, level, and supporting paragraphs in a human-readable format for manual annotation
    For each level of questions, write 'numq_perlevel' questions


    Parameters:
    -----------
    input_jsonl: Input data jsonl
    output_txt: Write output to this file for each ques level
    numq_perlevel: For each level, these many questions are written
    '''

    print("Reading dataset: {}".format(input_jsonl))
    qa_examples: List[Dict] = util.readJsonlDocs(input_jsonl)
    # Dict for mapping question level to list of questions of that level
    qlevel2qa = {}

    boolQuesIdxs = []

    for qidx, qaexample in enumerate(qa_examples):
        qlevel = qaexample[constants.qlevel_field]
        answer = qaexample[constants.ans_field]
        if answer in ["yes", "no"]:
            boolQuesIdxs.append(qidx)
        if qlevel not in qlevel2qa:
            qlevel2qa[qlevel] = []
        qlevel2qa[qlevel].append(qaexample)


    qlevel2sizes = dict([(ql, len(qas)) for ql, qas in qlevel2qa.items()])
    print("All question stats:")
    print(qlevel2sizes)

    # Shuffle questions
    for qlevel, qas in qlevel2qa.items():
        random.shuffle(qas)

    numqas_written = 0

    with open(output_txt, 'w') as outf:
        ''' TO write boolean questions only '''
        for qidx in boolQuesIdxs:
            qa = qa_examples[qidx]
            outstr = _getOutputStringForQA(qa)
            outf.write(outstr)
            numqas_written += 1


        ''' To write questions of all levels
        for qlevel, qas in qlevel2qa.items():
            for i in range(numq_perlevel):
                qa = qas[i]
                outstr = _getOutputStringForQA(qa)
                outf.write(outstr)
                numqas_written += 1
        '''

    print(f"QAs written = {numqas_written}")


def main(args):
    writeQAStrings(input_jsonl=args.input_jsonl, output_txt=args.output_txt, numq_perlevel=args.numq_perlevel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_txt', required=True)
    parser.add_argument('--numq_perlevel', default=50)

    args = parser.parse_args()

    main(args)

