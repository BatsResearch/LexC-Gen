
def nusax_convert_str_to_int_label(example):
    sentiment_labels = {"negative": 0, "neutral": 1, "positive": 2}
    example["label"] = sentiment_labels[example["label"].lower()]
    return example

def nusax_convert_int_to_str_label(L):
    sentiment_labels = {0: "negative", 1:"neutral", 2:"positive"}
    L = [sentiment_labels[e] for e in L]
    return L

def nusax_tokenize(tokenizer, examples):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    return examples.map(tokenize_function, batched=True)

    