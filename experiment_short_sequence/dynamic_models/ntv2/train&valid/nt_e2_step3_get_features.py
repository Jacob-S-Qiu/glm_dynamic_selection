# Now that we have the best model from the previous step and could directly extract the feature information of the model
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import os

tokenizer = AutoTokenizer.from_pretrained("pre_trained_model", trust_remote_code=True)

tokenized_test = torch.load('data/tokenized_test.pt')
tokenized_test = Dataset.from_dict(tokenized_test)

model_checkpoint = 'my_checkpoints'
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, trust_remote_code=True, num_labels=4)

test_trimmed_data = torch.load('data/test.pt')
sequences = test_trimmed_data['sequences']

labels = tokenized_test['labels']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)

results = {
    'sequences': sequences,
    'labels': labels,
    'feature': [],
    'confidence': [],
    'prediction': []
}

with torch.no_grad():
    for idx, batch in enumerate(tokenized_test):
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items() if k != 'labels'}  

        outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1].squeeze(0)
        feature = hidden_states.mean(dim=0).cpu().numpy() 
        
        # logits output of the classification header (for obtaining confidence)
        logits = outputs.logits.squeeze(0)
        softmax_scores = torch.softmax(logits, dim=-1)
        confidence, prediction = torch.max(softmax_scores, dim=-1)
        
        results['feature'].append(feature)
        results['confidence'].append(confidence.item())
        results['prediction'].append(prediction.item())

output_path = "nt_e2_step3_result.pt"
torch.save(results, output_path)
print(f"Results saved to {output_path}")