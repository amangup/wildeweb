from accelerate import Accelerator
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline

import datetime
import wandb
import os
import numpy as np
import pandas as pd
import evaluate
import torch


DATASET_NAME = "amang1802/wildeweb_cls_1M"
BERT_MODEL = "bert-base-cased"

load_dotenv()
os.environ["WANDB_PROJECT"]="soft_skills_classifier_bert_base"


def even_sampling(dataset):
    df = pd.DataFrame(dataset)

    # Get count of the least represented label
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()

    # Create balanced dataset by sampling equal numbers from each label
    balanced_df = pd.DataFrame()
    for label in range(len(label_counts)):
        label_subset = df[df['label'] == label].sample(min_count, random_state=42)
        balanced_df = pd.concat([balanced_df, label_subset])

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert back to Huggingface dataset
    balanced_dataset = Dataset.from_pandas(balanced_df)

    print(f"Original distribution: {label_counts}")
    print(f"New distribution: {balanced_df['label'].value_counts()}")

    return balanced_dataset

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    wandb.login()

    dataset = load_dataset(DATASET_NAME)['train'].select(range(1000_000))
    dataset = dataset.map(lambda score: {"label": max(0, score-1)}, input_columns=["classification_score"])
    dataset = even_sampling(dataset)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
    tokenized_dataset = dataset.map(lambda rows: tokenize_function(tokenizer, rows), batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    
    train_ds, test_ds = split_dataset['train'], split_dataset['test']

    print(f"{train_ds[0]['text']}\n\nLabel: {train_ds[0]['label']}")
    print(train_ds.unique('label'))

    accelerator = Accelerator()
    device = accelerator.device

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5, torch_dtype="auto").to(device)
    model.add_module('score', model.classifier)

    metric = evaluate.load("accuracy")
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1.0,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        output_dir=f"bert_base-{time_str}",
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        save_safetensors=False,
    )

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