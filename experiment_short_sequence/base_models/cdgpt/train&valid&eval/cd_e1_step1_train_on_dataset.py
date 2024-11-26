import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from config import get_config
from model import CDGPTSequencePrediction
from tokenizer import SentencePieceTokenizer

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

def train_one_epoch(model, train_loader, valid_loader, optimizer, criterion, device, epoch, accumulation_steps, save_every_n_epochs=None):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    step = 0

    checkpoint_dir = "../../lanyun-tmp/C10_cdgpt_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    steps_per_epoch = len(train_loader)
    validate_steps = int(steps_per_epoch * save_every_n_epochs)
    
    print(f"\n=== Start {epoch + 1} epoch ===")

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)["output"]
        loss = criterion(outputs, labels)
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item() * accumulation_steps:.4f}, LR: {current_lr:.6f}")

        total_loss += loss.item() * accumulation_steps  
        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        if (step + 1) % validate_steps * 5 == 0:
            print('start to valid the model')
            val_loss, val_f1, val_acc, val_precision, val_recall = validate(model, valid_loader, criterion, device)
            print(f"\nValidation at Step {step + 1}/{steps_per_epoch * epoch}")
            print(f"Validation Loss: {val_loss:.4f} | F1 Score: {val_f1:.4f} | Accuracy: {val_acc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}\n")
        
        if validate_steps and (step + 1) % validate_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}_{step + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
        step += 1
    
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    avg_loss = total_loss / len(train_loader)

    print(f"Average Loss: {avg_loss:.4f} | F1 Score: {f1:.4f} | Accuracy: {accuracy:.4f}\n")

    return avg_loss, f1, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)["output"]
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    avg_loss = total_loss / len(dataloader)

    return avg_loss, f1, accuracy, precision, recall


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration and Settings
cfg = get_config()
cfg.model.num_classes = 2
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
batch_size = 1
max_length = 1024  # Set according to the maximum length accepted by the model
learning_rate = 1e-5
epochs = 3
accumulation_steps=50
save_every_n_epochs = 0.2 
save_every_epoch = True 

train_dataset = SequenceDataset(train_file, tokenizer, max_length)
valid_dataset = SequenceDataset(valid_file, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training process
for epoch in range(epochs):
    train_loss, train_f1, train_acc = train_one_epoch(
        model, train_loader, valid_loader, optimizer, criterion, device, epoch, accumulation_steps, save_every_n_epochs
    )








