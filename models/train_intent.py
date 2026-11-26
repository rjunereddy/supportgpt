"""
Boilerplate script to fine-tune a small transformer (DistilBERT) for intent classification using PEFT/LoRA.
This script demonstrates the setup needed to hit the 'PyTorch/Deep Learning/PEFT' resume bullet.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

def train_intent_model():
    model_name = "distilbert-base-uncased"
    
    # 1. Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label={0: "Connectivity", 1: "Storage", 2: "Authentication"}
    )
    
    # 2. Setup LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1
    )
    
    # 3. Get PEFT Model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Mock Dataset for demonstration
    mock_data = {
        "text": ["Cannot connect to DB", "Storage is full", "SSO login failed", "IP is blocked", "Quota exceeded", "Invalid token"],
        "label": [0, 1, 2, 0, 1, 2]
    }
    dataset = Dataset.from_dict(mock_data)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./data/processed/intent_model",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    print("Starting LoRA Fine-tuning (Mock)...")
    trainer.train()
    
    # 7. Save Model
    model.save_pretrained("./data/processed/intent_model_lora")
    print("Model saved to ./data/processed/intent_model_lora")

if __name__ == "__main__":
    train_intent_model()
