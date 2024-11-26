# We've trained our checkpoints in step1, using checkpoints to verify all checkpoints in our checkpoints folder (don't forget to copy esm_model.py file to each checkpoint folder).
import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

tokenizer = AutoTokenizer.from_pretrained("pre_trained_model", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("pre_trained_model", trust_remote_code=True, num_labels=4)

tokenized_test = torch.load('data/tokenized_test.pt')
tokenized_test = Dataset.from_dict(tokenized_test)

soft_labels_valid = torch.load('data/merged_soft_label_and_models_prediction_test_dataset.pt')
hy_predictions = soft_labels_valid['hy_prediction']
nt_predictions = soft_labels_valid['nt_prediction']
cd_predictions = soft_labels_valid['cd_prediction']
nt_predictions_reversed = [(1 - p) for p in nt_predictions]  # Reverse ntv2 predictions

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    max_indices = np.argmax(predictions, axis=-1) 

    final_predictions = []
    for idx, max_index in enumerate(max_indices):
        if max_index == 0:
            final_pred = hy_predictions[idx]
        elif max_index == 1:
            final_pred = nt_predictions[idx]
        elif max_index == 2:
            final_pred = cd_predictions[idx]
        else:
            # Category 4 case: ntv2 predictions are reversed
            final_pred = nt_predictions_reversed[idx]
        final_predictions.append(final_pred)

    accuracy = accuracy_score(labels, final_predictions)
    f1 = f1_score(labels, final_predictions, average='binary')
    precision = precision_score(labels, final_predictions, average='binary')
    recall = recall_score(labels, final_predictions, average='binary')

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

checkpoint_dir = 'my_checkpoints'
checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint')]

results = []

for checkpoint_path in checkpoints:
    print(f"Evaluating model from {checkpoint_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=4, trust_remote_code=True)
    
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
