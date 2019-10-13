import os
import sys
import json
from _jsonnet import evaluate_file

# Jsonnet file to evaluate
params_file = sys.argv[1]

# Filepath of output file
outfile = sys.argv[2]

# Environment variables
ext_vars = dict(os.environ)

file_dict = json.loads(evaluate_file(params_file, ext_vars=ext_vars))

with open(outfile, "w") as handle:
    json.dump(file_dict, handle, indent=4)
