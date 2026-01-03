*2. `train.py`**
```python
#!/usr/bin/env python3
"""
Fine-tuning script for German Legal Summarizer.
Uses LoRA (Low-Rank Adaptation) for efficient training.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json
import argparse
from typing import Dict, List
import evaluate
import numpy as np

def load_dataset(data_path: str) -> Dataset:
    """Load and prepare dataset from JSONL file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Convert to Hugging Face Dataset format
    dataset_dict = {
        "text": [item["text"] for item in data],
        "summary": [item["summary"] for item in data]
    }
    
    return Dataset.from_dict(dataset_dict)

def preprocess_function(examples, tokenizer, max_input_length: int, max_target_length: int):
    """Tokenize and preprocess examples."""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    """Compute ROUGE metrics for evaluation."""
    rouge = evaluate.load("rouge")
    
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extract median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune German Legal Summarizer")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn",
                       help="Base model to fine-tune")
    parser.add_argument("--dataset_path", type=str, default="data/train_dataset.jsonl",
                       help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./legal-summarizer-model",
                       help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_input_length", type=int, default=1024,
                       help="Maximum input length")
    parser.add_argument("--max_target_length", type=int, default=256,
                       help="Maximum target length")
    
    args = parser.parse_args()
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    
    # Split dataset (80/20 train/validation)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, args.max_input_length, args.max_target_length
        ),
        batched=True
    )
    
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, args.max_input_length, args.max_target_length
        ),
        batched=True
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        report_to="none"  # Change to "wandb" if using Weights & Biases
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
