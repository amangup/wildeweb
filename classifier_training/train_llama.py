from accelerate import Accelerator
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, setup_chat_format

import datetime
import wandb
import os
import numpy as np
import pandas as pd
import evaluate
import torch

DATASET_NAME = "amang1802/wildeweb_cls_labels_v1"
MODEL_NAME = "Llama-3.2-1B"
MODEL_ID = f"meta-llama/{MODEL_NAME}"
MAX_TEXT_LEN = 12000

INSTRUCTIONS = """
Below is an extract from a web page. You are an AI content evaluator focused on assessing educational material's value for soft skills development. Soft skills include conversational ability, empathy, leadership skills, public speaking, confidence building, critical thinking, problem solving, professional writing, teamwork, digital literacy, professional attitude, work ethic, career management and intercultural fluency. 

You will analyze content using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the extract hows superficial coverage of basic communication and teamwork concepts without meaningful depth or practical application. Professional development opportunities are limited to theoretical knowledge, and problem-solving scenarios lack complexity or real-world context. Cultural awareness and digital literacy elements are either absent or extremely basic.
- Add another point if the extract specifically includes discussion of soft skills and includes straightforward communication scenarios and simple team dynamics, but lacks nuanced interaction or complex problem-solving opportunities. Professional development focuses on fundamental skills with limited practical application, while cultural awareness and digital literacy are present but superficial.
- Award a third point if the extract specifically includes discussion of soft skills andfeatures realistic scenarios that integrate emotional intelligence, leadership challenges, and critical thinking opportunities. Professional development includes practical applications with meaningful context, while incorporating cultural awareness and modern digital literacy skills throughout the material. 
- Grant a fourth point if the extract specifically includes discussion of soft skills and presents complex scenarios requiring sophisticated communication, strategic thinking, and advanced problem-solving across multiple contexts. Professional development opportunities are comprehensive and practical, with strong emphasis on intercultural fluency and technological adaptation.
- Bestow a fifth point if the extract specifically includes discussion of soft skills and seamlessly integrates advanced communication, leadership, and problem-solving scenarios that mirror real-world complexity. Professional development opportunities span multiple contexts with sophisticated cultural awareness, while digital literacy and practical application are woven throughout every element.

Output either 1,2,3,4 or 5.
"""

load_dotenv()
os.environ["WANDB_PROJECT"] = f"soft_skills_classifier"

def chat_messages(tokenizer, examples):
    truncated_texts = [text[:MAX_TEXT_LEN] for text in examples[:text]]

    messages = [[
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": text + "\n\nScore:"},
        {"role": "assistant", "content": f"{label+1}"}
    ] for text, label in zip(truncated_texts, examples['label'])]
    
    return {"messages": messages}


def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    wandb.login()

    dataset = load_dataset(DATASET_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
    chat_formatted_ds = dataset.map(lambda rows: chat_messages(tokenizer, rows), batched=True)
    train_ds, test_ds = chat_formatted_ds['train'], chat_formatted_ds['test']

    print(train_ds[0]['messages'])
    print(train_ds.unique('label'))

    accelerator = Accelerator()
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
    model, tokenizer = setup_chat_format(model, tokenizer)

    metric = evaluate.load("accuracy")
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    training_args = SFTConfig(
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=2.0,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        output_dir=f"{MODEL_NAME}-{time_str}",
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        save_safetensors=False,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=lambda eval_pred: compute_metrics(metric, eval_pred),
    )

    trainer.train()

if __name__ == "__main__":
    main()