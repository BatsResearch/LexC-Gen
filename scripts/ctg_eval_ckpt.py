from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch
from trl import SFTTrainer
import argparse
import pathlib
import random
import collections

from utils.lexicon_usage_check import lexicon_usage
import stanza

from accelerate import Accelerator

lang = "en"
lang_tokenizer = stanza.Pipeline(lang, processors='tokenize')

device_index = Accelerator().process_index
device_map = {"": device_index}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bigscience/bloomz-7b1", help="model name")
parser.add_argument("--peft_model_id", type=str, required=True, help="peft model id")
parser.add_argument("--tgt_lang", type=str, required=True, help="target language")
parser.add_argument("--lexicons_dir", type=str, required=True, help="lexicons directory")
parser.add_argument("--ctg_source_lang", type=str, default="English", help="source language")
parser.add_argument("--task_data", type=str, choices=['nusax', 'sib200'], help="nusax (sentiment analysis) or sib200 (topic classification)")
parser.add_argument("--max_new_tokens", type=int, default=256, help="max new tokens")
parser.add_argument("--total", type=int, default=10000, help="number of generated examples")
parser.add_argument("--top_p", type=float, default=.1, help="top p")
parser.add_argument("--temperature", type=float, default=1., help="temperature")
parser.add_argument("--do_print_output", action="store_true", help="print output")
args = parser.parse_args()

base_model = args.model
peft_model_id = args.peft_model_id
tgt_lang = args.tgt_lang

# load lexicons
lexicons = collections.defaultdict(list)
lexicon_file = f"{args.lexicons_dir}/en_{tgt_lang}.tsv"
with open(lexicon_file) as rf:
    for line in rf:
        line = line.strip().split('\t')
        if ' ' in line[0]:
            continue
        lexicons[line[0]].append(line[1])

# Load base model
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map, # https://github.com/huggingface/accelerate/issues/1840
        cache_dir="/users/zyong2/data/zyong2/huggingface"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'left' # change to left now
tokenizer.pad_token = tokenizer.eos_token

# instruction
if args.task_data == "nusax":
    instruction = "We are creating dataset for {language} sentiment analysis using a provided set of tokens. Tokens: {list_tokens} {label} sentence:"
elif args.task_data == "sib200":
    instruction = "We are creating dataset for {language} topic classification using a provided set of tokens. Tokens: {list_tokens} Sentence related to {label}:"
else:
    raise ValueError(f"Unsupported task data: {args.task_data}")

# load adapter and merge
inference_model = PeftModel.from_pretrained(model, peft_model_id)
inference_model = inference_model.merge_and_unload()

lexicon_usages = list()
total = args.total
while len(lexicon_usages) < total:
    # generation
    tokens = random.choices(list(lexicons.keys()), k=10)
    if args.task_data == "nusax":
        label = random.choice(["Positive", "Neutral", "Negative"])
    elif args.task_data == "sib200":
        label = random.choice(["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"])
    else:
        raise ValueError(f"Unsupported task data: {args.task_data}")

    input_text = instruction.format(list_tokens=str(tokens), label=label, language=args.ctg_source_lang)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    output = inference_model.generate(**inputs,
                                      do_sample=True, 
                                      top_p=args.top_p, 
                                      temperature=args.temperature,
                                      max_new_tokens=args.max_new_tokens)
    output_txt = tokenizer.decode(output[0], skip_special_tokens=False)

    #### extract output
    s = ''
    try:
        if args.task_data == "nusax":
            split_template = f"] {label} sentence:"
        elif args.task_data == "sib200":
            split_template = f"] Sentence related to {label}:"
        else:
            raise ValueError(f"Unsupported task data: {args.task_data}")

        if split_template in output_txt:
            _, s = output_txt.split(split_template)
    except:
        continue

    s = s.strip()
    if not s or not s.endswith("</s>"):
        continue
    s = s.replace("</s>", "")
    
    #### tokenize output
    txt_tokens = [word.text for sent in lang_tokenizer(s).sentences for word in sent.words]
    tokenized_text = " ".join(txt_tokens)

    #### count lexicon usage
    cur_lexicon_usage = lexicon_usage(tokenized_text, lexicon_list=tokens)
    lexicon_usages.append(cur_lexicon_usage)

    if args.do_print_output:
        print(tokens)
        print(f"{label} sentence (use {cur_lexicon_usage} provided tokens): {tokenized_text}")

print('-'*50, flush=True)
avg_lexicon_usage = sum(lexicon_usages) / len(lexicon_usages)
print(f"Model {peft_model_id} has average lexicon usage {avg_lexicon_usage}")
print('-'*50, flush=True)
