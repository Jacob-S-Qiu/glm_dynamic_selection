from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, load_from_disk
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

checkpoint = 'pre_trained_model'
max_length = 20_000
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=4,  
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

def tokenize_function(examples):
    return tokenizer(examples["sequences"], padding="max_length", max_length=max_length, truncation=True)

if os.path.exists('data/processed_test_dataset'):
    tds = load_from_disk('data/processed_test_dataset')
    tds = tds.remove_columns(['true_label', 'hy_prediction', 'nt_prediction', 'cd_prediction'])
    print("Loaded processed test dataset from disk.")
else:
    test_data = torch.load('data/merged_soft_label_and_models_prediction_test_dataset.pt')
    test_sequences = test_data['sequence']
    test_soft_labels = test_data['soft_labels']
    test_labels = test_data['label']
    hy_predictions = test_data['hy_prediction']
    nt_predictions = test_data['nt_prediction']
    cd_predictions = test_data['cd_prediction']
    tds = Dataset.from_dict({
        'sequences': test_sequences,
        'labels': test_soft_labels,
        'true_label': test_labels,
        'hy_prediction': hy_predictions,
        'nt_prediction': nt_predictions,
        'cd_prediction': cd_predictions
    })
    tds = tds.map(tokenize_function, batched=True, num_proc=15)
    tds = tds.remove_columns(['sequences'])
    tds.set_format("torch")
    tds.save_to_disk('data/processed_test_dataset')
    print("Processed test dataset and saved to disk.")

tds = tds.remove_columns(['true_label', 'hy_prediction', 'nt_prediction', 'cd_prediction'])

def custom_compute_metrics(pred):
    soft_labels = pred.label_ids  
    predictions = pred.predictions 
    model_predictions = predictions.argmax(axis=-1)  
    valid_data = torch.load("data/merged_soft_label_and_models_prediction_test_dataset.pt")
    true_labels = valid_data['label']
    hy_predictions = valid_data['hy_prediction']
    nt_predictions = valid_data['nt_prediction']
    cd_predictions = valid_data['cd_prediction']

    all_preds = []
    all_labels = []
    
    for i, model_index in enumerate(model_predictions):
        true_label = true_labels[i]
        if model_index == 0:  # Select the Hyena model
            pred_label = hy_predictions[i]
        elif model_index == 1:  # Select the NTv2 model
            pred_label = nt_predictions[i]
        elif model_index == 2:  # Select the CDGPT model
            pred_label = cd_predictions[i]
        elif model_index == 3:  # In the fourth case, select the inverse result of Hyena
            pred_label = 1 - hy_predictions[i]
        else:
            raise ValueError("Invalid model index")

        all_preds.append(pred_label)
        all_labels.append(true_label)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

checkpoint_dir = 'my_checkpoints'
checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint')]

results = []

for checkpoint_path in checkpoints:
    print(f"Evaluating model from {checkpoint_path}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=4, trust_remote_code=True)
    model.eval()
    training_args = TrainingArguments(
        output_dir="tmp",
        per_device_eval_batch_size=16,  
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tds, 
        compute_metrics=custom_compute_metrics,
        tokenizer=tokenizer
    )
    eval_result = trainer.evaluate()
    print(eval_result)
    eval_result['checkpoint'] = checkpoint_path
    results.append(eval_result)

print(eval_result)
