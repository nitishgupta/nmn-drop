from typing import List, Dict, Union
import random
import json
import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")

def get_sentence_graph(doc):
    edges = []
    for token in doc:
        tokenidx = token.i
        for child in token.children:
            childidx = child.i
            edges.append(('{}'.format(tokenidx),
                          '{}'.format(childidx)))
    graph = nx.Graph(edges)
    return graph

def shortest_dependency_path(graph, e1: int, e2: int):
    try:
        shortest_path = nx.shortest_path(graph, source=str(e1), target=str(e2))
    except nx.NetworkXNoPath:
        shortest_path = []
    return shortest_path

text = "In mid-June 1940, when international attention was focused on the German invasion of France, Soviet NKVD troops raided border posts in Lithuania, Estonia and Latvia."
doc = nlp(text)
sents = [sent for sent in doc.sents]

def get_token_idx(charoffset, spacy_doc):
    tokenidx = None
    for tidx, token in enumerate(spacy_doc):
        # charoffset is inclusive; so match starting from first char
        if token.idx <= charoffset < token.idx + len(token.text):
            return tidx

    return None


dt = doc.text[5:10]
import pdb
pdb.set_trace()

graph = get_sentence_graph(doc)
# shortpath = shortest_dependency_path(sents[0], "Soviet", "Lithuania")
# print(shortpath)
# shortpath = shortest_dependency_path(sents[0], "Soviet", "France")
# print(shortpath)

tokens = [f"{token}-{token.i}" for token in doc]
print(tokens)
shortpath = shortest_dependency_path(graph, 18, 16)
print(shortpath)
shortpath = shortest_dependency_path(graph, 18, 25)
print(shortpath)






