from utils import util, spacyutils
import json

nlp = spacyutils.getSpacyNLP()
# nlpwhite = spacyutils.getWhiteTokenizerSpacyNLP()

a = "Prince Yury Dolgorukov\u00a0 on October 11"

d = nlp(a)

# Non-breaking character token contains space in it
tokens = [t for t in d]

# \xa0 is a separate token with space - ['Prince', 'Yury', 'Dolgorukov', '\xa0 ', 'on', 'October', '11']
tokenstext = [t.text for t in tokens]

# This is false
print(f"Are all tokens without space: {all([len(t.split(' ')) == 1 for t in tokenstext])}")