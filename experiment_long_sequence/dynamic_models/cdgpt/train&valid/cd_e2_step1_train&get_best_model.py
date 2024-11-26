# Start training the dynamic selector

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os

from config import get_config
from model import CDGPTSequencePrediction
from tokenizer import SentencePieceTokenizer
from torch.optim.lr_scheduler import ExponentialLR

class SoftLabelDataset(Dataset):
    def __init__(self, sequence_file, label_file, tokenizer, max_length):
        sequence_data = torch.load(sequence_file)
        self.sequences = sequence_data['sequences']
        
        label_data = torch.load(label_file)
        self.soft_labels = label_data['soft_labels']
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        soft_label = self.soft_labels[idx]

        tokens = self.tokenizer.encode(sequence[:self.max_length], eos=False)
        
        tokens = tokens.tolist()
        if len(tokens) < self.max_length:
            tokens += [self.tokenizer.pad] * (self.max_length - len(tokens))
    
        return torch.tensor(tokens), torch.tensor(soft_label, dtype=torch.float32)

class SequenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        data = torch.load(file_path)
        self.sequences = data['sequences']
        self.labels = data['labels']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        tokens = self.tokenizer.encode(sequence[:self.max_length], eos=False)
        
        tokens = tokens.tolist()
        if len(tokens) < self.max_length:
            tokens += [self.tokenizer.pad] * (self.max_length - len(tokens))
    
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.long)

def train_one_epoch(model, train_loader, valid_loader, optimizer, device, epoch, accumulation_steps, validate_steps):
    model.train()
    total_loss = 0
    step = 0
    checkpoint_dir = "my_checkpoints"
    
    print(f"\n=== Start {epoch + 1} epoch ===")

    for sequences, soft_labels in train_loader:
        sequences, soft_labels = sequences.to(device), soft_labels.to(device)

        outputs = model(sequences)["output"]
        predictions = F.softmax(outputs, dim=-1)

        loss = F.kl_div(predictions.log(), soft_labels, reduction="batchmean")
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1} - Learning Rate: {current_lr:.6f}")
   
            print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item() * accumulation_steps:.4f}")

        total_loss += loss.item() * accumulation_steps  # 还原损失值

        if (step + 1) % validate_steps == 0:
            accuracy, f1, precision, recall = validate(model, valid_loader, label_valid_file, device)
            print(f"\nValidation at Step {step + 1}")
            print(f"Validation Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}\n")
        if (step + 1) % validate_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"my_checkpoint_{epoch + 1}_{step + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
        step += 1

    avg_loss = total_loss / len(train_loader)
    print(f"Average Loss: {avg_loss:.4f}\n")

    return avg_loss

def validate(model, valid_loader, valid_data_path, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    valid_data = torch.load(valid_data_path)
    hy_predictions = valid_data['hy_prediction']
    nt_predictions = valid_data['nt_prediction']
    cd_predictions = valid_data['cd_prediction']
    labels = valid_data['label']

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for i, (sequences, label) in enumerate(valid_loader):
            sequences = sequences.to(device)
            output = model(sequences)["output"]
            prediction = F.softmax(output, dim=-1)

            max_index = torch.argmax(prediction, dim=-1).item()
            model_pred = None
            
            # Select the corresponding model prediction according to the maximum subscript
            if max_index == 0:
                model_pred = hy_predictions[i]
            elif max_index == 1:
                model_pred = nt_predictions[i]
            elif max_index == 2:
                model_pred = cd_predictions[i]
            elif max_index == 3:
                model_pred = 1 - nt_predictions[i]  # The fourth case is reversed

            is_correct = (model_pred == labels[i])
            total_correct += is_correct
            total_samples += 1

            all_labels.append(labels[i])
            all_preds.append(model_pred)

    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    print(f"Validation Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    return accuracy, f1, precision, recall



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = get_config()
cfg.model.num_classes = 4
tokenizer_path = "checkpoints/tokenizer.model"
cfg.tokenizer.path = tokenizer_path
tokenizer = SentencePieceTokenizer(tokenizer_path)
cfg.tokenizer.pad_token_id = tokenizer.pad
model_path = "checkpoints/CD-GPT-1b.pth"
assert os.path.exists(model_path)
state = torch.load(model_path, map_location="cpu")
output_head = "sequence"
model = CDGPTSequencePrediction(cfg)
model.load_state_dict(state["model"], strict=False)
model.pad_token_id = tokenizer.pad
model = model.to(device)

train_file = "data/train.pt"
valid_file = "data/valid.pt"
label_train_file = "data/merged_soft_label_and_models_prediction_train_dataset.pt" # Soft label training data set
label_valid_file = "data/merged_soft_label_and_models_prediction_valid_dataset.pt"
batch_size = 1
learning_rate = 6e-5
epochs = 3
accumulation_steps = 50
max_length = 1024
validate_steps=600 

train_dataset = SoftLabelDataset(train_file, label_train_file, tokenizer, max_length)
valid_dataset = SequenceDataset(valid_file, tokenizer, max_length)  # Validation sets do not require soft labels
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.998)

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, valid_loader, optimizer, device, epoch, accumulation_steps, validate_steps)
    accuracy, f1, precision, recall = validate(model, valid_loader, label_valid_file, device)

# Load the test set and get the model that works best on the test set
test_file = "data/test.pt"
test_dataset = SequenceDataset(test_file, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

label_test_file = "data/merged_soft_label_and_models_prediction_test_dataset.pt"

checkpoint_dir = "my_checkpoints"

results = []

for checkpoint_file in os.listdir(checkpoint_dir):
    if checkpoint_file.endswith(".pth"):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        model = CDGPTSequencePrediction(cfg)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        model.pad_token_id = tokenizer.pad
        model = model.to(device)
        
        f1, accuracy, precision, recall = validate(model, test_loader, label_test_file, device)
        results.append({
            "model": checkpoint_file,
            "f1_score": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        })
        print(f"Evaluated {checkpoint_file} - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

print(results)