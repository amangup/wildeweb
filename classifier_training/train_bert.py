from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline

import numpy as np
import evaluate
import torch

DATASET_NAME = "amang1802/wildeweb_cls_1M"
BERT_MODEL = "bert-base-cased"


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    dataset = load_dataset(DATASET_NAME)['train'].select(range(10000))
    dataset = dataset.map(lambda score: {"label": max(0, score-1)}, input_columns=["classification_score"])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
    tokenized_dataset = dataset.map(lambda rows: tokenize_function(tokenizer, rows), batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    
    train_ds, test_ds = split_dataset['train'], split_dataset['test']

    print(f"{train_ds[0]['text']}\n\nLabel: {train_ds[0]['label']}")
    print(train_ds.unique('label'))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5, torch_dtype="auto").to(device)

    metric = evaluate.load("accuracy")

    training_args = TrainingArguments(output_dir="bert_test1", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
    )

    trainer.train()

if __name__ == "__main__":
    main()