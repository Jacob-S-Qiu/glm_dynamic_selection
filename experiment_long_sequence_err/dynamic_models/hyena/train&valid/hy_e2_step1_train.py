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

if os.path.exists('data/processed_train_dataset'):
    tds = load_from_disk('data/processed_train_dataset')
    print("Loaded processed training dataset from disk.")
else:
    train_data = torch.load('data/merged_soft_label_and_models_prediction_train_dataset.pt')
    train_sequences = train_data['sequence']
    train_soft_labels = train_data['soft_labels']
    tds = Dataset.from_dict({'sequences': train_sequences, 'labels': train_soft_labels})
    tds = tds.map(tokenize_function, batched=True, num_proc=15)
    tds = tds.remove_columns(['sequences'])
    tds.set_format("torch")
    tds.save_to_disk('data/processed_train_dataset')
    print("Processed training dataset and saved to disk.")
if os.path.exists('data/processed_valid_dataset'):
    vds = load_from_disk('data/processed_valid_dataset')
    print("Loaded processed validation dataset from disk.")
else:
    valid_data = torch.load('data/merged_soft_label_and_models_prediction_valid_dataset.pt')
    valid_sequences = valid_data['sequence']
    valid_soft_labels = valid_data['soft_labels']
    valid_labels = valid_data['label']
    hy_predictions = valid_data['hy_prediction']
    nt_predictions = valid_data['nt_prediction']
    cd_predictions = valid_data['cd_prediction']
    vds = Dataset.from_dict({
        'sequences': valid_sequences,
        'labels': valid_soft_labels,
        'true_label': valid_labels,
        'hy_prediction': hy_predictions,
        'nt_prediction': nt_predictions,
        'cd_prediction': cd_predictions
    })
    vds = vds.map(tokenize_function, batched=True, num_proc=15)
    vds = vds.remove_columns(['sequences'])
    vds.set_format("torch")
    vds.save_to_disk('data/processed_valid_dataset')
    print("Processed validation dataset and saved to disk.")

# Custom Trainer for using KL divergence as a loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(torch.float32)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # KL divergence loss is calculated
        loss = torch.nn.functional.kl_div(log_probs, labels, reduction='batchmean')
        if return_outputs:
            return loss, outputs
        else:
            return loss

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

training_args = TrainingArguments(
    output_dir="my_checkpoints",
    num_train_epochs=4,
    per_device_train_batch_size=28,  
    gradient_accumulation_steps=2,
    gradient_checkpointing=True, 
    learning_rate=1.5e-5,
    save_strategy="steps",
    logging_steps=30,
    save_steps=30,  
    metric_for_best_model="f1",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tds,
    eval_dataset=vds,  
    compute_metrics=custom_compute_metrics
)

trainer.train()





