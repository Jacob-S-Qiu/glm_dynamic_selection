import torch
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import Trainer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

tokenized_train = torch.load('data/tokenized_train.pt')
tokenized_valid = torch.load('data/tokenized_valid.pt')

train_soft_labels_data = torch.load('data/merged_soft_label_and_models_prediction_train_dataset.pt')
valid_soft_labels_data = torch.load('data/merged_soft_label_and_models_prediction_valid_dataset.pt')

tokenized_train['labels'] = train_soft_labels_data['soft_labels']
tokenized_valid['labels'] = valid_soft_labels_data['soft_labels']

tokenized_train = Dataset.from_dict(tokenized_train)
tokenized_valid = Dataset.from_dict(tokenized_valid)


tokenizer = AutoTokenizer.from_pretrained("../C2/pre-model-NTv2", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "pre_trained_model",
    trust_remote_code=True,
    num_labels=4 
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        log_probs = F.log_softmax(logits, dim=-1)
        labels = labels.type_as(log_probs)
        
        # KL divergence loss is calculated
        loss = F.kl_div(log_probs, labels, reduction='batchmean')

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, _ = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    merged_data = torch.load('data/merged_soft_label_and_models_prediction_valid_dataset.pt')
    labels = merged_data['label']  
    hy_predictions = merged_data['hy_prediction']
    nt_predictions = merged_data['nt_prediction']
    cd_predictions = merged_data['cd_prediction']
    hyena_predictions = merged_data['hy_prediction']  # invert

    final_predictions = []
    for idx, pred in enumerate(preds):
        if pred == 0:
            model_pred = hy_predictions[idx]
        elif pred == 1:
            model_pred = nt_predictions[idx]
        elif pred == 2:
            model_pred = cd_predictions[idx]
        elif pred == 3:
            model_pred = 1 - hyena_predictions[idx]
        else:
            model_pred = 0

        final_predictions.append(model_pred)

    final_predictions = np.array(final_predictions)
    labels = np.array(labels)

    if len(final_predictions) != len(labels):
        print(f"Inconsistent lengths: predictions={len(final_predictions)}, labels={len(labels)}")
        raise ValueError("Predictions and labels have inconsistent lengths")

    accuracy = accuracy_score(labels, final_predictions)
    f1 = f1_score(labels, final_predictions)
    precision = precision_score(labels, final_predictions)
    recall = recall_score(labels, final_predictions)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

args = TrainingArguments(
    output_dir="my_checkpoints",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    learning_rate=7e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=40,
    num_train_epochs=2,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="f1_score",
    label_names=["labels"],
)

optimizer = AdamW(model.parameters(), lr=7e-5)
scheduler = ExponentialLR(optimizer, gamma=0.997)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

train_results = trainer.train()
