"""
Adapted from https://github.com/cindyxinyiwang/expand-via-lexicon-based-adaptation/blob/main/src/make_pseudo_mono.py
"""
import random
import sys
from tqdm import tqdm
import collections

from pathlib import Path
import json

def word_translate(tokenized_sents:list, lexicon:dict, max_sent_count:int=float('inf'), show_tqdm:bool=False):
    total_non_translated = total_translated = 0
    translated_data = []
    src_tokens = collections.Counter() # track the number of times each token appears in the source corpora
    seen_src_lexicon_words = set()
    seen_tgt_lexicon_words = set()

    line_count = 0
    for line in tqdm(tokenized_sents, desc="Translating...", disable=not show_tqdm):
        toks = line.strip().split()
        new_toks = []
        _translated = _non_translated = 0
        for t in toks:
            src_tokens[t] += 1
            if t in lexicon:
                new_t = random.choice(list(lexicon[t]))
                new_toks.append(new_t)
                _translated += 1
                seen_src_lexicon_words.add(t)
                seen_tgt_lexicon_words.add(new_t)
            else:
                new_toks.append(t)
                _non_translated += 1

        # if _translated >= 3: # this is the one causing differences in the total number of EN tokens.
        #     # if more than 3 tokens replaced in the sentence, then accept the sentence
        # in pseudo-labeled, we don't have threshold
        translated_data.append(" ".join(new_toks))
        total_translated += _translated
        total_non_translated += _non_translated
            
        line_count += 1
        if line_count >= max_sent_count: break
    
    all_src_lexicon_words = set(lexicon.keys())
    all_tgt_lexicon_words = set()
    for k, v in lexicon.items():
        all_tgt_lexicon_words.update(v)

    out = {
        "translated_data": translated_data,
        "total_non_translated": total_non_translated,
        "total_translated": total_translated,
        "total_tokens": total_translated + total_non_translated,
        "src_tokens_counter": src_tokens,
        "seen_src_lexicon_words": len(seen_src_lexicon_words),
        "seen_tgt_lexicon_words": len(seen_tgt_lexicon_words),
        "total_src_lexicon_size": len(all_src_lexicon_words),
        "total_tgt_lexicon_size": len(all_tgt_lexicon_words),
        "src_utilization": len(seen_src_lexicon_words) / len(all_src_lexicon_words),
        "tgt_utilization": len(seen_tgt_lexicon_words) / len(all_tgt_lexicon_words),
    }
    
    return out