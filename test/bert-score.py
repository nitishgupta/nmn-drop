from bert_score import BERTScorer

def print_scores(scorer, cands, refs):
    bert_out = scorer.score(cands, refs)
    precision, recall, fscores = bert_out[0], bert_out[1], bert_out[2]
    for i in range(len(refs)):
        cand = cands[i]
        ref = refs[i]
        p, r, f1 = precision[i], recall[i], fscores[i]
        print(f"{cand} \t {ref} \t {f1}")


# scorer = BERTScorer(model_type="bert-base-uncased", rescale_with_baseline=True, lang="en")
scorer = BERTScorer(model_type="bert-base-uncased", rescale_with_baseline=False)

refs = open("test/refs.txt", 'r').readlines()
cands = open("test/hyps.txt", 'r').readlines()

refs = [x.strip() for x in refs]
cands = [x.strip() for x in cands]

import pdb
pdb.set_trace()