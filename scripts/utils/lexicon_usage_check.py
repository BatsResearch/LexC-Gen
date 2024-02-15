import random
import sys
from tqdm import tqdm
import collections

from pathlib import Path
import json

def lexicon_usage(tokenized_sent:str, lexicon_list:list[str]):
    # count how many words in lexicon is used in the sentence
    toks = tokenized_sent.strip().split()
    lexicon = set(lexicon_list)
    seen_lex = set()
    for t in toks:
        if t in lexicon:
            seen_lex.add(t)
    return len(seen_lex)