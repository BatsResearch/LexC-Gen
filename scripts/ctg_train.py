# from https://www.datacamp.com/tutorial/mistral-7b-tutorial
# from https://huggingface.co/docs/trl/main/en/sft_trainer

from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch
from datasets import load_dataset
from trl import SFTTrainer
import argparse

from accelerate import Accelerator

device_index = Accelerator().process_index
device_map = {"": device_index}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bigscience/bloomz-7b1", help="model name")
parser.add_argument("--local-rank", type=int, default=-1, help="local rank for distributed training")
parser.add_argument("--ctg_dataset", type=str, required=True, help="CTG training dataset (txt file)")
parser.add_argument("--output_dir", type=str, required=True, help="output directory for CTG-trained model")

### hyperparameters
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--max_seq_length", type=int, default=1024, help="max sequence length")
parser.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup ratio")
parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay")
parser.add_argument("--num_train_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--max_steps", type=int, default=-1, help="max steps")
parser.add_argument("--logging_steps", type=int, default=1000, help="max sequence length")
parser.add_argument("--eval_steps", type=int, default=1000, help="eval steps")
parser.add_argument("--save_steps", type=int, default=1000, help="save steps")
parser.add_argument("--logging_strategy", type=str, default="steps", help="logging strategy")
args = parser.parse_args()

base_model = args.model
txt_dataset_name = args.ctg_dataset # may have to explore lemmatized dataset
output_dir = args.output_dir

# load dataset
dataset = load_dataset('text', data_files={"train": txt_dataset_name}, split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.05, seed=42)

# Load base model
bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map, # https://github.com/huggingface/accelerate/issues/1840
        cache_dir="/users/zyong2/data/zyong2/huggingface" # TODO: remove this line
)
model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
# tokenizer.add_bos_token, tokenizer.add_eos_token = True, True # TODO: doesn't work for BLOOMZ

# Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
target_modules = {
    'bigscience/bloomz-7b1': ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
}
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules[base_model],
)
model = get_peft_model(model, peft_config)


########### Training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="steps",
    save_steps=args.save_steps,
    logging_strategy="steps",
    logging_steps=args.logging_steps,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=args.max_steps,
    warmup_ratio=args.warmup_ratio,
    group_by_length=True,
    lr_scheduler_type="constant",
    ddp_find_unused_parameters=False, # https://github.com/mymusise/ChatGLM-Tuning/issues/152#issuecomment-1581989852
    local_rank=args.local_rank,
)


########### Trainer
class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

class SFTTrainerWrapper(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # (wrapper) debugging check train set
        print("debugging check trainset")
        print(self.tokenizer.decode(self.train_dataset[0]['input_ids'], skip_special_tokens=False)) # first example

# Setting sft parameters
callbacks = [PeftSavingCallback()]
trainer = SFTTrainerWrapper(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    max_seq_length=args.max_seq_length,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    callbacks=callbacks,
)

trainer.train()
