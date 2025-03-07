import argparse
import time
from collections import Counter
import numpy as np
from datasets import load_dataset
from vllm import LLM


DATASET_ID = "amang1802/wildeweb_cls_labels_v1"  # Replace with your dataset
TOKENIZER_ID = "bert-base-cased"  # Replace with your tokenizer


def classify_batch(examples, model):
    outputs = model.classify(examples["text"])
    predictions = [np.argmax(output.outputs.probs) for output in outputs]
    return {"prediction": predictions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    dataset = load_dataset(DATASET_ID, split="test")
    
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=2,
        task="classify"
    )
    
    start_time = time.time()
    
    processed_dataset = dataset.map(
        lambda examples: classify_batch(examples, model),
        batched=True,
        batch_size=args.batch_size
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Get all predictions and true labels
    predictions = processed_dataset["prediction"]
    true_labels = processed_dataset["label"]
    total_samples = len(predictions)
    samples_per_second = total_samples / execution_time
    
    # Calculate accuracy
    correct = sum(p == l for p, l in zip(predictions, true_labels))
    accuracy = correct / total_samples * 100
    
    # Count label occurrences
    label_counts = Counter(predictions)
    
    print(f"Processed {total_samples} samples in {execution_time:.2f} seconds")
    print(f"Inference speed: {samples_per_second:.2f} samples/sec")
    print(f"Accuracy: {accuracy:.2f}%")
    

    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count}")



if __name__ == "__main__":
    main()