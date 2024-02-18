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
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--train_dataset_config", type=str, default="english", required=True)
parser.add_argument("--output_dir", type=str, default="/users/zyong2/data/zyong2/scaling/data/processed/901-hf-sa/reprod_cls/")
parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
parser.add_argument("--cache_dir", type=str, default="/users/zyong2/data/bats/models")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--early_stopping_patience", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--num_seeds", type=int, default=5)

args = parser.parse_args()

train_dataset_config = args.train_dataset_config
train_dataset = load_dataset("BatsResearch/NusaX-senti-LexC-Gen", train_dataset_config, split="train")
val_dataset = load_dataset("BatsResearch/NusaX-senti-LexC-Gen", train_dataset_config, split="validation")
test_dataset = load_dataset("indonlp/NusaX-senti", args.lang, split="test")

#### Tokenize data
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

print(tokenized_train_dataset)
print(tokenized_val_dataset)
print(tokenized_test_dataset)

############# Train Model #############
output_dir = pathlib.Path(args.output_dir) / f"{train_dataset_config}-{args.model_name.split('/')[-1]}"
print(f"output_dir: {output_dir}")
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
