from typing import List, Tuple, Dict, Union, Callable
import os
import re
import json
import copy
import argparse
from collections import defaultdict

from allennlp.data import Token
from allennlp.data.tokenizers import SpacyTokenizer

from semqa.utils.qdmr_utils import read_drop_dataset, node_from_dict, Node, nested_expression_to_lisp, \
    read_json_dataset, read_jsonl
from utils import util, spacyutils


""" 
    Adds minimal project(select) w/o question-attention program-supervision to all Squad questions.
    The idea is that the NMN model will figure out the question-arguments itself.
"""





