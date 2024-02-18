
topic_labels = {
    "science/technology": 0,
    "travel": 1,
    "politics": 2,
    "sports": 3,
    "health": 4, 
    "entertainment": 5,
    "geography": 6
}

topic_labels_inversed = {v: k for k, v in topic_labels.items()}

def sib200_convert_str_to_int_label(example):
    example["label"] = topic_labels[example["category"]]
    return example

def sib200_convert_int_to_str_label(L):
    L = [topic_labels_inversed[e] for e in L]
    return L

def sib200_tokenize(tokenizer, examples, max_length):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    return examples.map(tokenize_function, batched=True)

    