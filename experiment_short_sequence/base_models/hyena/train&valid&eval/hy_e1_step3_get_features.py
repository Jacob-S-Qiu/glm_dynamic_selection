# Use the best model to extract the required feature content

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F

def extract_features_and_losses(model_path, data_path, tokenizer_checkpoint, output_path="results.pt", max_length=1600000, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, trust_remote_code=True
    )
    model.config.output_hidden_states = True  # Ensure hidden states are returned
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, trust_remote_code=True)
    
    data = torch.load(data_path)
    sequences, labels = data['sequences'], data['labels']
    print(f"Loaded {len(sequences)} sequences from {data_path}")

    tokens = tokenizer(
        sequences, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt"
    )

    input_ids = tokens['input_ids'].to(device)
    labels = torch.tensor(labels).to(device)

    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    results = {
        'feature': [],
        'label': [],
        'loss': [],
        'prediction': [],
        'confidence': []
    }
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_input_ids, batch_labels = batch
            
            if batch_idx % 300 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            outputs = model(
                input_ids=batch_input_ids, 
                labels=batch_labels
            )

            if outputs.hidden_states is not None:
                feature = outputs.hidden_states[-1][:, 0, :]

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            softmax_scores = F.softmax(outputs.logits, dim=-1)
            highest_confidence, _ = torch.max(softmax_scores, dim=-1)
            confidence_scores = highest_confidence.tolist()

            for feat, label, pred, conf in zip(feature, batch_labels, predictions, confidence_scores):
                results['feature'].append(feat.cpu().numpy())
                results['label'].append(label.item())
                results['loss'].append(outputs.loss.item())
                results['prediction'].append(pred.item())
                results['confidence'].append(conf) 

    accuracy = correct_predictions / total_samples
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    torch.save(results, output_path)
    print(f"Results saved to {output_path}")

    return results



import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F
from datasets import load_from_disk

def extract_features_and_losses_optimized(model_path, data_path, output_path="results.pt", batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, trust_remote_code=True
    )
    model.config.output_hidden_states = True  # Ensure hidden states are returned
    model.to(device)
    model.eval()

    test_dataset = load_from_disk(data_path) 
    print(f"Loaded preprocessed dataset from {data_path}")

    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    results = {
        'feature': [],
        'label': [],
        'loss': [],
        'prediction': [],
        'confidence': []
    }
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_input_ids = batch['input_ids'].to(device)
            batch_labels = batch['labels'].to(device)

            if batch_idx % 300 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            outputs = model(input_ids=batch_input_ids, labels=batch_labels)

            if outputs.hidden_states is not None:
                feature = outputs.hidden_states[-1][:, 0, :]

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            softmax_scores = F.softmax(outputs.logits, dim=-1)
            highest_confidence, _ = torch.max(softmax_scores, dim=-1)
            confidence_scores = highest_confidence.tolist()

            for feat, label, pred, conf in zip(feature, batch_labels, predictions, confidence_scores):
                results['feature'].append(feat.cpu().numpy())
                results['label'].append(label.item())
                results['loss'].append(outputs.loss.item())
                results['prediction'].append(pred.item())
                results['confidence'].append(conf)

    accuracy = correct_predictions / total_samples
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    torch.save(results, output_path)
    print(f"Results saved to {output_path}")

    return results



results = extract_features_and_losses(
    model_path="my_checkpoints/best_checkpoint", 
    data_path="data/test.pt", 
    tokenizer_checkpoint='pre_trained_model',
    output_path="hy_e1_result_test.pt"
)
