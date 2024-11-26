from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments,Trainer
from datasets import load_dataset, Dataset
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from sklearn.metrics import matthews_corrcoef, f1_score

base_checkpoints_path = "pre_trained_model"
tokenizer = AutoTokenizer.from_pretrained(base_checkpoints_path,trust_remote_code=True)
max_length = tokenizer.model_max_length
model = AutoModelForSequenceClassification.from_pretrained(base_checkpoints_path,trust_remote_code=True,num_labels=2)
config = AutoConfig.from_pretrained(base_checkpoints_path,trust_remote_code=True)
print(config)

tokenized_train=torch.load('data/tokenized_train.pt')
tokenized_valid=torch.load('data/tokenized_valid.pt')
tokenized_test=torch.load('data/tokenized_test.pt')

tokenized_train = Dataset.from_dict(tokenized_train)
tokenized_valid = Dataset.from_dict(tokenized_valid)
tokenized_test = Dataset.from_dict(tokenized_test)

args = TrainingArguments(
    output_dir="my_checkpoints",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=30,
    learning_rate=6e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size= 1,
    gradient_accumulation_steps= 40,
    num_train_epochs= 2,
    logging_steps= 100,
    label_names=["labels"],
)

optimizer = AdamW(model.parameters(), lr=6e-5)

scheduler = ExponentialLR(optimizer, gamma=0.999)

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

trainer = Trainer(
    model,
    args,
    train_dataset= tokenized_train,
    eval_dataset= tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

train_results = trainer.train()
