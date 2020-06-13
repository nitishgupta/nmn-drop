from typing import List, Dict
import json
from collections import defaultdict


def func(**kwargs):
    x = kwargs["x"]
    y = kwargs["y"]

    print(x)
    print(y)


if __name__=="__main__":
    func(**{"x": 5, "y": 10})