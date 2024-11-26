from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_from_disk
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

checkpoint = 'pre_trained_model'
max_length = 20_000
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

subset_ratio = 1

def tokenize_function(examples):
    return tokenizer(examples["sequences"], padding="max_length", max_length=max_length, truncation=True)

# Check and load the pre-processed training set
if os.path.exists('data/processed_train_dataset'):
    tds = load_from_disk('data/processed_train_dataset')
    print("Loaded processed training dataset from disk.")
else:
    train_data = torch.load('data/train.pt')
    train_sequences = train_data['sequences'][:int(len(train_data['sequences']) * subset_ratio)]
    train_labels = train_data['labels'][:int(len(train_data['labels']) * subset_ratio)]
    tds = Dataset.from_dict({'sequences': train_sequences, 'labels': train_labels})
    tds = tds.map(tokenize_function, batched=True, num_proc=4)  
    tds = tds.map(lambda x: {"labels": torch.tensor(x['labels'], dtype=torch.int64)}, batched=True)
    tds = tds.map(lambda x: {"input_ids": torch.tensor(x['input_ids'], dtype=torch.int64)}, batched=True)
    tds = tds.remove_columns(['sequences'])
    tds.set_format("torch")
    tds.save_to_disk('data/processed_train_dataset')
    print("Processed training dataset and saved to disk.")
    
if os.path.exists('data/processed_valid_dataset'):
    vds = load_from_disk('data/processed_valid_dataset')
    print("Loaded processed validation dataset from disk.")
else:
    valid_data = torch.load('data/valid.pt')
    valid_sequences = valid_data['sequences'][:int(len(valid_data['sequences']) * 0.01)]
    valid_labels = valid_data['labels'][:int(len(valid_data['labels']) * 0.01)]
    vds = Dataset.from_dict({'sequences': valid_sequences, 'labels': valid_labels})
    vds = vds.map(tokenize_function, batched=True, num_proc=4) 
    vds = vds.map(lambda x: {"labels": torch.tensor(x['labels'], dtype=torch.int64)}, batched=True)
    vds = vds.map(lambda x: {"input_ids": torch.tensor(x['input_ids'], dtype=torch.int64)}, batched=True)
    vds = vds.remove_columns(['sequences'])
    vds.set_format("torch")
    vds.save_to_disk('data/processed_valid_dataset')
    print("Processed validation dataset and saved to disk.")

if os.path.exists('data/processed_test_dataset'):
    vds = load_from_disk('data/processed_test_dataset')
    print("Loaded processed test dataset from disk.")
else:
    test_data = torch.load('data/test.pt')
    test_sequences = test_data['sequences'][:int(len(test_data['sequences']) * 0.01)]
    test_labels = test_data['labels'][:int(len(test_data['labels']) * 0.01)]
    tds = Dataset.from_dict({'sequences': test_sequences, 'labels': test_labels})
    tds = tds.map(tokenize_function, batched=True, num_proc=15)  
    tds = tds.map(lambda x: {"labels": torch.tensor(x['labels'], dtype=torch.int64)}, batched=True)
    tds = tds.map(lambda x: {"input_ids": torch.tensor(x['input_ids'], dtype=torch.int64)}, batched=True)
    tds = tds.remove_columns(['sequences'])
    tds.set_format("torch")
    tds.save_to_disk('data/processed_test_dataset')
    print("Processed test dataset and saved to disk.")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir="my_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16, 
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tds,
    eval_dataset=vds, 
    compute_metrics=compute_metrics 
)

# Start training
result = trainer.train()
print(result)

















