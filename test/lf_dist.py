from typing import List, Dict, Union
import random
import json

depparse_jsonl = "/shared/nitishg/data/squad/squad-dev-v1.1_questions_depparse.jsonl"

examples = []
with open(depparse_jsonl) as f:
    for line in f:
        examples.append(json.loads(line))

import pdb
pdb.set_trace()



