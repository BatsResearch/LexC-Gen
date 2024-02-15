import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--existing_task_data", type=str, required=True, help="existing task data")
parser.add_argument("--output_file", type=str, required=True, help="output txt file for CTG training")
parser.add_argument("--task_data", type=str, choices=['nusax', 'sib200'], help="nusax (sentiment analysis) or sib200 (topic classification)")

args = parser.parse_args()

language = "English"
FORMAT = "<s>{instruct} {answer}</s>" # needs to add bos and eos tokens because BLOOMZ in current HF doesn't support add_bos_token and add_eos_token

input_rf = pathlib.Path(args.existing_task_data)
output_file = pathlib.Path(args.output_file)
output_file.parent.mkdir(parents=True, exist_ok=True) # create parent directory if not exists

wf = open(output_file, "w+")

if input_rf.suffix == ".csv":
    df = pd.read_csv(input_rf)
elif input_rf.suffix == ".tsv":
    df = pd.read_csv(input_rf, sep="\t")
else:
    raise ValueError(f"Unsupported existing task data format: {input_rf.suffix}")

if args.task_data == "nusax":
    texts = df['text'].tolist()
    labels = df['label'].tolist()
elif args.task_data == "sib200":
    texts = df['text'].tolist()
    labels = df['category'].tolist()
else:
    raise ValueError(f"Unsupported task data: {args.task_data}")


for text, label in zip(texts, labels):
    toks = text.split()

    # skip short sentences
    if len(toks) <= 10:
        continue

    while True:
        num_tokens_sampled = int(np.random.normal(len(toks), 1))
        if num_tokens_sampled > 0 and num_tokens_sampled < len(toks): break
    sampled_toks = random.sample(toks, num_tokens_sampled)

    ### preprocessing
    # remove punctuations
    sampled_toks = [tok for tok in sampled_toks if tok.isalpha()]
    
    # remove duplicates
    sampled_toks = list(set(sampled_toks))

    ##### generate dataset
    if args.task_data == "nusax":
        instruct = f"We are creating dataset for {language} sentiment analysis using a provided set of tokens. Tokens: {sampled_toks} {label.capitalize()} sentence:"
    elif args.task_data == "sib200":
        instruct = f"We are creating dataset for English topic classification using a provided set of tokens. Tokens: {sampled_toks} Sentence related to {label.lower()}:"
    else:
        raise ValueError(f"Unsupported task data: {args.task_data}")

    input_sentence = FORMAT.format(instruct=instruct, answer=text)
    print(input_sentence)
    print(text, label)
    wf.write(input_sentence)
    wf.write("\n")

            
            
