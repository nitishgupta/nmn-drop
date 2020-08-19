import bert_score

refs = open("test/refs.txt", 'r').readlines()
cands = open("test/hyps.txt", 'r').readlines()

refs = [x.strip() for x in refs]
cands = [x.strip() for x in cands]

# print(refs)
# print(cands)

bert_out = bert_score.score(cands, refs, lang="en")


precision, recall, fscores = bert_out[0], bert_out[1], bert_out[2]

for i in range(len(refs)):
    ref = refs[i]
    hyp = cands[i]
    p, r, f1 = precision[i], recall[i], fscores[i]
    print(f"{ref} \t {hyp} \t {f1}")

# print(bert_out)

