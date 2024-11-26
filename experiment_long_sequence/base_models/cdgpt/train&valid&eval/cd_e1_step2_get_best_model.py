# In the previous step, we trained a bunch of checkpoints, so we can check each one to find the one that works best

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

        # Tokenize and truncate
        tokens = self.tokenizer.encode(sequence[:self.max_length], eos=False)
        
        tokens = tokens.tolist()
        if len(tokens) < self.max_length:
            tokens += [self.tokenizer.pad] * (self.max_length - len(tokens))
    
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.long)

def validate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)["output"]
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    return f1, accuracy, precision, recall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = get_config()
cfg.model.num_classes = 2
tokenizer_path = "checkpoints/tokenizer.model"
cfg.tokenizer.path = tokenizer_path
tokenizer = SentencePieceTokenizer(tokenizer_path)
cfg.tokenizer.pad_token_id = tokenizer.pad
max_length = 1024 

test_file = "data/test.pt"
test_dataset = SequenceDataset(test_file, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

checkpoint_dir = "my_checkpoints"

# Record model results
results = []

# Iterate over all saved models and validate
for checkpoint_file in os.listdir(checkpoint_dir):
    if checkpoint_file.endswith(".pth"):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        model = CDGPTSequencePrediction(cfg)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        model.pad_token_id = tokenizer.pad
        model = model.to(device)
        
        f1, accuracy, precision, recall = validate(model, test_loader, device)
        results.append({
            "model": checkpoint_file,
            "f1_score": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        })
        print(f"Evaluated {checkpoint_file} - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")






