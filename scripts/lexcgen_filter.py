import pathlib
import pandas as pd
import argparse
import yaml

import numpy as np
import evaluate
import pandas as pd

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from utils.nusax_post_lexcgen import nusax_convert_str_to_int_label, nusax_convert_int_to_str_label, nusax_tokenize
from utils.sib200_post_lexcgen import sib200_convert_str_to_int_label, sib200_convert_int_to_str_label, sib200_tokenize


parser = argparse.ArgumentParser()
parser.add_argument("--target_lang", type=str, required=True, help="target low-resource language")
parser.add_argument("--file", type=str, required=True, help="path to the LexC-Gen generated file")
parser.add_argument("--task_data", type=str, choices=['nusax', 'sib200'], help="nusax (sentiment analysis) or sib200 (topic classification)")
parser.add_argument("--output_dir", type=str, required=True, help="output directory")
parser.add_argument("--filter_config_file", type=str, default="./scripts/cls_configs/filter.yaml")
parser.add_argument("--filter_model_name", type=str, default="bert-base-multilingual-cased", help="model name to use for filtering")
parser.add_argument("--filter_train_file", type=str, required=True, help="path to the training file for the filter model")
parser.add_argument("--filter_valid_file", type=str, required=True, help="path to the validation file for the filter model")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--cache_dir", type=str, default="/users/zyong2/data/bats/models") # TODO: remove 
parser.add_argument("--no_filter", action="store_true", help="if set, no filtering will be done. So we simply relabel the data (i.e., label distillation).")
args = parser.parse_args()

file = pathlib.Path(args.file)
if not file.exists():
    raise FileNotFoundError(f"File not found: {file}")
output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

################### Convert to CSV/TSV ###################
labels = list()
texts = list()
ids = list()
with open(file) as rf:
    for i, line in enumerate(rf):
        s = ""
        try:
            if args.task_data == "nusax":
                for label in ["Positive", "Negative", "Neutral"]:
                    template = f"{label} sentence:"
                    if template in line:
                        _, s = line.split(template)
                        break
            elif args.task_data == "sib200":
                for label in ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]:
                    template = f"Sentence related to {label}:"
                    if template in line:
                        _, s = line.split(template)
                        break
            else:
                raise ValueError(f"Unsupported task data: {args.task_data}")
        except Exception as e:
            # this catches weird generations such as neural text degeneration (repeated texts) and so on
            continue

        s = s.strip()
        if not s.endswith("</s>"):
            continue
        labels.append(label)
        texts.append(s.strip("</s>"))
        ids.append(i)

if args.task_data == "nusax":
    df = pd.DataFrame.from_dict({'id': ids, 'text': texts, 'label': labels})
    df.to_csv(output_dir / f"{file.stem}.csv", index=False)
elif args.task_data == "sib200":
    df = pd.DataFrame.from_dict({'id': ids, 'text': texts, 'category': labels})
    df.to_csv(output_dir / f"{file.stem}.tsv", sep="\t", index=False)
else:
    raise ValueError(f"Unsupported task data: {args.task_data}")

################### Filter out bad examples ###################
print("Filtering out bad examples...")
with open(args.filter_config_file, 'r') as filter_config_file:
    filter_config = yaml.safe_load(filter_config_file)[args.task_data]
tokenizer = AutoTokenizer.from_pretrained(args.filter_model_name, cache_dir=args.cache_dir)

if args.task_data == "nusax":
    train_dataset = load_dataset("csv", data_files=args.filter_train_file)['train']
    val_dataset = load_dataset("csv", data_files=args.filter_valid_file)['train']
    test_dataset = Dataset.from_pandas(df) # our generated dataset

    train_dataset = train_dataset.map(nusax_convert_str_to_int_label)
    val_dataset = val_dataset.map(nusax_convert_str_to_int_label)
    test_dataset = test_dataset.map(nusax_convert_str_to_int_label)

    tokenized_train_dataset = nusax_tokenize(tokenizer, train_dataset)
    tokenized_val_dataset = nusax_tokenize(tokenizer, val_dataset)
    tokenized_test_dataset = nusax_tokenize(tokenizer, test_dataset)

elif args.task_data == "sib200":
    train_dataset_df = pd.DataFrame(pd.read_csv(args.filter_train_file, sep="\t"))
    train_dataset = Dataset.from_pandas(train_dataset_df)

    val_dataset_df = pd.DataFrame(pd.read_csv(args.filter_valid_file, sep="\t"))
    val_dataset = Dataset.from_pandas(val_dataset_df)

    test_dataset = Dataset.from_pandas(df)

    train_dataset = train_dataset.map(sib200_convert_str_to_int_label)
    val_dataset = val_dataset.map(sib200_convert_str_to_int_label)
    test_dataset = test_dataset.map(sib200_convert_str_to_int_label)

    tokenized_train_dataset = sib200_tokenize(tokenizer, train_dataset, max_length=filter_config['max_length'])
    tokenized_val_dataset = sib200_tokenize(tokenizer, val_dataset, max_length=filter_config['max_length'])
    tokenized_test_dataset = sib200_tokenize(tokenizer, test_dataset, max_length=filter_config['max_length'])

print(tokenized_train_dataset)
print(tokenized_val_dataset)
print(tokenized_test_dataset)

output_dir = pathlib.Path(args.output_dir)
# adding target_lang just to avoid clashing with other experiments
# cls is only trained on English data
cls_dir = output_dir / f"tmp/{args.task_data}-cls-{args.target_lang}-{args.filter_model_name}"
cls_dir.mkdir(parents=True, exist_ok=True)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(args.filter_model_name, cache_dir=args.cache_dir, num_labels=int(filter_config['num_labels']))
training_args = TrainingArguments(
    output_dir=cls_dir,
    num_train_epochs=int(filter_config['num_epochs']),
    per_device_train_batch_size=int(filter_config['batch_size']),
    per_device_eval_batch_size=int(filter_config['batch_size']),
    learning_rate=float(filter_config['lr']),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model='eval_accuracy',
    load_best_model_at_end=True,
    seed=args.seed,
    data_seed=args.seed,
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=filter_config['early_stopping_patience'])],
)

trainer.train()

res = trainer.predict(tokenized_test_dataset)
predictions = np.argmax(res.predictions, axis=-1)
if args.task_data == "nusax":
    predictions = nusax_convert_int_to_str_label(predictions)
elif args.task_data == "sib200":
    predictions = sib200_convert_int_to_str_label(predictions)

texts = df['text']
labels = df[filter_config['label_column']]
filtered_texts = list()
filtered_labels = list()
filtered_ids = list()
for i, (text, label) in enumerate(zip(texts,labels)):
    if args.no_filter:
        filtered_texts.append(text)
        filtered_labels.append(label)
        filtered_ids.append(i)
        continue

    if label.lower() == predictions[i].lower():
        filtered_texts.append(text)
        filtered_labels.append(label)
        filtered_ids.append(i)

filtered_df = pd.DataFrame.from_dict({"id": filtered_ids, "text":filtered_texts, filter_config['label_column']: filtered_labels})
if args.task_data == "nusax":
    filtered_df.to_csv(output_dir / f"filtered-{file.stem}.csv", index=False)
    print(f"Filtered dataset saved to {output_dir / f'filtered-{file.stem}.csv'}")
elif args.task_data == "sib200":
    filtered_df.to_csv(output_dir / f"filtered-{file.stem}.tsv", sep="\t", index=False)
    print(f"Filtered dataset saved to {output_dir / f'filtered-{file.stem}.tsv'}")
