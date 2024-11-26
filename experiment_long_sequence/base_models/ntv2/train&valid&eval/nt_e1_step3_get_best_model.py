# The trained model is verified uniformly and the one with the best effect is obtained

import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import load_dataset, Dataset

base_checkpoints_path = "pre_trained_model"
tokenizer = AutoTokenizer.from_pretrained(base_checkpoints_path,trust_remote_code=True)
max_length = tokenizer.model_max_length
model = AutoModelForSequenceClassification.from_pretrained(base_checkpoints_path,trust_remote_code=True,num_labels=2)

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids

    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions)
    precision = precision_score(references, predictions)
    recall = recall_score(references, predictions)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

# Traverse all checkpoint folders in the directory
checkpoint_dir = 'my_checkpoints'
checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint')]

tokenized_test=torch.load('data/tokenized_test.pt')
tokenized_test = Dataset.from_dict(tokenized_test)
print(len(tokenized_test['labels']))

results = []

for checkpoint_path in checkpoints:
    print(f"Evaluating model from {checkpoint_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2, trust_remote_code=True)
    
    training_args = TrainingArguments(
        output_dir="check_points",
        per_device_eval_batch_size=20,  
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test, 
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    eval_result = trainer.evaluate()
    
    eval_result['checkpoint'] = checkpoint_path
    results.append(eval_result)
    print(eval_result)