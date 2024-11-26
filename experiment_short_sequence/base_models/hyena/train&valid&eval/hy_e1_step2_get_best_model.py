# Test the checkpoints tested in step1 checkpoints to find the most effective checkpoints

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

checkpoint = 'pre_trained_model'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
test_dataset = load_from_disk('data/processed_test_dataset')  # 加载预处理后的测试集

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

checkpoint_dir = 'my_checkpoints'
checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint')]
checkpoints

results = []

for checkpoint_path in checkpoints:
    print(f"Evaluating model from {checkpoint_path}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2, trust_remote_code=True)
    
    training_args = TrainingArguments(
        output_dir="tmp",
        per_device_eval_batch_size=16,  
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset, 
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    eval_result = trainer.evaluate()

    eval_result['checkpoint'] = checkpoint_path
    results.append(eval_result)

print(results)