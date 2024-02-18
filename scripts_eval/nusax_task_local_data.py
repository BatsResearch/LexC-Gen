import pathlib
import argparse

import numpy as np
import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer


parser = argparse.ArgumentParser()
parser.add_argument("--target_lang", type=str, required=True)
parser.add_argument("--train_csv_path", type=str, required=True)
parser.add_argument("--valid_csv_path", type=str, required=True)
parser.add_argument("--test_csv_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./tmp")
parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
parser.add_argument("--cache_dir", type=str, default="/users/zyong2/data/bats/models")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--early_stopping_patience", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--num_seeds", type=int, default=5)

args = parser.parse_args()

train_dataset = load_dataset("csv", data_files=str(args.train_csv_path))
val_dataset = load_dataset("csv", data_files=str(args.valid_csv_path))
test_dataset = load_dataset("csv", data_files=str(args.test_csv_path))

#### Tokenize data
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

def convert_str_to_int_label(example):
    sentiment_labels = {"negative": 0, "neutral": 1, "positive": 2}
    example["label"] = sentiment_labels[example["label"].lower()]
    return example

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)

train_dataset = train_dataset.map(convert_str_to_int_label)
val_dataset = val_dataset.map(convert_str_to_int_label)
test_dataset = test_dataset.map(convert_str_to_int_label)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)['train']
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)['train']
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)['train']

print(tokenized_train_dataset)
print(tokenized_val_dataset)
print(tokenized_test_dataset)

############# Train Model #############
output_dir = pathlib.Path(args.output_dir) / f"eval-nusax-{args.target_lang}-{args.model_name.split('/')[-1]}"
print(f"task classifiers saved to output_dir: {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

accs = []
for seed in range(args.num_seeds):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, cache_dir=args.cache_dir, num_labels=3)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model='eval_accuracy',
        load_best_model_at_end=True,
        seed=seed,
        data_seed=seed,
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    ############# Test Model #############
    res = trainer.predict(tokenized_test_dataset)

    print(f"---- Test Accuracy (seed {seed}) ----")
    acc = res.metrics['test_accuracy']
    print(acc)
    print("-"*45)

    accs.append(acc)
    #######################################

print("="*45)
print("==== Average Test Accuracy ====")
print(f"{np.mean(accs)*100:.1f} ± {np.std(accs)*100:.1f}")
print("="*45)
