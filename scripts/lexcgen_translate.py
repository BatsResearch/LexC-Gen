import stanza
import pandas as pd
import pathlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stanza_lang", type=str, default="en")
parser.add_argument("--file", type=str, required=True, help="path to the LexC-Gen generated file")
parser.add_argument("--task_data", type=str, choices=['nusax', 'sib200'], help="nusax (sentiment analysis) or sib200 (topic classification)")
args = parser.parse_args()

lang_tokenizer = stanza.Pipeline(args.stanza_lang, processors='tokenize')
data_file = pathlib.Path(file)
df = pd.read_csv(data_file)

texts = df['text']
tokenized_texts = list()
for text in texts:
    txt_tokens = [word.text for sent in lang_tokenizer(text).sentences for word in sent.words]
    tokenized_texts.append(" ".join(txt_tokens))


# df['text'] = tokenized_texts
# save_file = f"/oscar/data/sbach/zyong2/scaling/data/processed/306-ctg-final-postpoc/tokenized_{data_file.stem}.csv"
# df.to_csv(save_file, index=False)
# print(f"File saved to {save_file}")
