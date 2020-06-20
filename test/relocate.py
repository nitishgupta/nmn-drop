from typing import List, Dict, Optional, Tuple
import json


_labels = {0: 'O', 1: 'I'}

def convert_tags_to_spans(token_tags: List[str], labels_scheme: str = "BIO") -> List[Tuple[int, int]]:
    spans = []
    prev = 'O'
    current = []
    for tokenidx, tag in enumerate(token_tags):
        if tag == labels['I']:
            if labels_scheme == "BIO":
                if prev == "B" or prev == "I":
                    current.append(tokenidx)
                    prev = "I"
                else:
                    # Illegal I, treat it as O
                    prev = "O"
            elif labels_scheme == "IO":
                if prev == "I":
                    current.append(tokenidx)  # continue span
                else:
                    if current:
                        spans.append((current[0], current[-1]))
                    current = [tokenidx]
                    prev = "I"
            else:
                raise NotImplementedError

        if tag == "O":
            if prev == "O":
                continue
            elif prev == "B" or prev == "I":
                if current:
                    spans.append((current[0], current[-1]))
                current = []
                prev = "O"
        if tag == "B":
            if current:
                spans.append((current[0], current[-1]))
            current = [tokenidx]
            prev = "B"

    if current:
        # residual span
        spans.append((current[0], current[-1]))

    return spans