import json
import datasets.hotpotqa.utils.constants as constants

f = open('/srv/local/data/nitishg/data/hotpotqa/raw/hotpot_train_v1.json', 'r')

json_objs = json.load(f)

f.close()

outf = open('/srv/local/data/nitishg/data/hotpotqa/raw/sample.json', 'w')

relevant = []

for o in json_objs:
    ques = o[constants.q_field]
    if ques.find('Horia') > -1:
        relevant.append(o)

json.dump(relevant, outf)
outf.close()
