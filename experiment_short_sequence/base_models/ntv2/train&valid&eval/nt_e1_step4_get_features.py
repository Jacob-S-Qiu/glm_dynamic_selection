# Extract the features of the model hidden layer output from the best models

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

def extract_features_and_losses_with_features(checkpoint_path, data_path, output_path, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2, trust_remote_code=True)
    model.to(device)
    model.eval() 

    tokenized_test = torch.load(data_path)
    input_ids = tokenized_test['input_ids'].to(device)
    attention_mask = tokenized_test['attention_mask'].to(device)
    labels = tokenized_test['labels'].to(device)
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    results = {
        'label': [],
        'loss': [],
        'prediction': [],
        'confidence': [],
        'feature': [] 
    }
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch_input_ids, batch_attention_mask, batch_labels = batch
            
            if batch_idx % 500 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            outputs = model(
                input_ids=batch_input_ids, 
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                output_hidden_states=True 
            )

            hidden_states = outputs.hidden_states[-1] 
            features = hidden_states.mean(dim=1) 

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            softmax_scores = F.softmax(outputs.logits, dim=-1)
            highest_confidence, _ = torch.max(softmax_scores, dim=-1)
            confidence_scores = highest_confidence.tolist()

            for label, pred, conf, feat in zip(batch_labels, predictions, confidence_scores, features):
                results['label'].append(label.item())
                results['loss'].append(outputs.loss.item())
                results['prediction'].append(pred.item())
                results['confidence'].append(conf)
                results['feature'].append(feat.cpu().numpy())

    accuracy = correct_predictions / total_samples
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    torch.save(results, output_path)
    print(f"Results saved to {output_path}")

    return results

results = extract_features_and_losses_with_features(
    checkpoint_path="my_checkpoints/best_checkpoint", 
    data_path="data/tokenized_test.pt", 
    output_path="confidence_index/nt_e1_result_test.pt"
)