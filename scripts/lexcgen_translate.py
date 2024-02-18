import stanza
import pandas as pd
import pathlib
import collections

import argparse
from tqdm import tqdm
from utils.word_translate import word_translate

parser = argparse.ArgumentParser()
parser.add_argument("--stanza_lang", type=str, default="en")
parser.add_argument("--file", type=str, required=True, help="path to the LexC-Gen filtered and generated file")
parser.add_argument("--target_lang", type=str, required=True, help="target low-resource language (use Gatitos language code)")
parser.add_argument("--lexicons_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

lang_tokenizer = stanza.Pipeline(args.stanza_lang, processors='tokenize')
data_file = pathlib.Path(args.file)
if not data_file.exists():
    raise FileNotFoundError(f"File not found: {data_file}")

if data_file.suffix == ".csv":
    df = pd.read_csv(data_file)
elif data_file.suffix == ".tsv":
    df = pd.read_csv(data_file, sep='\t')
else:
    raise ValueError(f"File format not supported: {data_file.suffix}")

### Load lexicon
lexicon_file = f"{args.lexicons_dir}/en_{args.target_lang}.tsv"
lexicon_translate = collections.defaultdict(set)
with open(lexicon_file, 'r') as f:
    lexicon = f.readlines()
    lexicon = [line.strip() for line in lexicon]
    lexicon = [line.split('\t') for line in lexicon]
    for line in lexicon:
        lexicon_translate[line[0]].add(line[1])

### Tokenize
texts = df['text']
tokenized_texts = list()
for text in tqdm(texts, desc="Tokenizing..."):
    txt_tokens = [word.text for sent in lang_tokenizer(text).sentences for word in sent.words]
    tokenized_texts.append(" ".join(txt_tokens))

### Translate
_res = word_translate(tokenized_texts, lexicon_translate)
translated_texts = _res['translated_data']
df['text'] = translated_texts

output_dir = pathlib.Path(args.output_dir)
print(f"Saving translated file to {output_dir}")
if not output_dir.exists():
    output_dir.mkdir(parents=True)
if data_file.suffix == ".csv":
    print(data_file)
    save_file = output_dir / f"translated_{data_file.stem}.csv"
    df.to_csv(save_file, index=False)
    print(f"File saved to {save_file}")
elif data_file.suffix == ".tsv":
    save_file = output_dir / f"translated_{data_file.stem}.tsv"
    df.to_csv(save_file, sep='\t', index=False)
    print(f"File saved to {save_file}")

### Print stats
print("-"*30)
print("Language:", args.target_lang)
print(f"LexC-Gen-{args.target_lang} total tokens:", _res['total_tokens'])
print(f"LexC-Gen-{args.target_lang} translated portion: {_res['total_translated'] / _res['total_tokens']*100:.1f}%")
print(f"LexC-Gen-{args.target_lang} src lexicon-utilization: {_res['src_utilization']*100:0.1f}%")
print(f"LexC-Gen-{args.target_lang} tgt lexicon-utilization: {_res['tgt_utilization']*100:0.1f}%")
print("-"*30)
