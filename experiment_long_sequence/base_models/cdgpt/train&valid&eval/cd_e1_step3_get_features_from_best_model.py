# The previous step obtained the model that worked best. Now use that model to extract the features
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from config import get_config
from model import CDGPTSequencePrediction
from tokenizer import SentencePieceTokenizer
from tqdm import tqdm
import torch.nn.functional as F

def extract_model_outputs_with_features(model, data_loader, criterion, device):
    # Initialize a results dictionary with a new field for features
    results = {
        'label': [],
        'loss': [],
        'prediction': [],
        'confidence': [],
        'feature': [] 
    }
    correct_predictions = 0
    total_samples = 0
    total_loss = 0 

    # Storage for intermediate features
    features = []

    def hook_fn(module, input, output):
        features.append(output.detach().cpu()) 

    hook_handle = model.cls_head.dense.register_forward_hook(hook_fn) 
    model.eval()

    with torch.no_grad():
        for batch_idx, (batch_input_sequences, batch_labels) in enumerate(tqdm(data_loader)):
            batch_input_sequences = batch_input_sequences.to(device)
            batch_labels = batch_labels.to(device)

            features.clear()

            outputs = model(batch_input_sequences)["output"]

            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() 

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
            softmax_scores = F.softmax(outputs, dim=-1)
            highest_confidence, _ = torch.max(softmax_scores, dim=-1)
            confidence_scores = highest_confidence.tolist()

            for label, pred, conf, feat in zip(batch_labels, predictions, confidence_scores, features):
                results['label'].append(label.item())
                results['loss'].append(loss.item())
                results['prediction'].append(pred.item())
                results['confidence'].append(conf)
                results['feature'].append(feat.squeeze(0).numpy())  # Remove batch dimension and convert to numpy

    accuracy = correct_predictions / total_samples
    avg_loss = total_loss / total_samples
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")

    # Remove the hook
    hook_handle.remove()

    return results

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_config()
    cfg.model.num_classes = 2
    tokenizer_path = "checkpoints/tokenizer.model"
    cfg.tokenizer.path = tokenizer_path
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    cfg.tokenizer.pad_token_id = tokenizer.pad
    max_length = 1024

    checkpoint_path = 'my_checkpoints/best_model.pt'
    model = CDGPTSequencePrediction(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.pad_token_id = tokenizer.pad
    model = model.to(device)

    valid_file = "data/test.pt"
    valid_dataset = SequenceDataset(valid_file, tokenizer, max_length)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    output_path = "confidence_index/cd_e1_result_test.pt"
    results = extract_model_outputs_with_features(model, valid_loader, criterion, device)

    torch.save(results, output_path)
    print(f"Results saved to {output_path}")