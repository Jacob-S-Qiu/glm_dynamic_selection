# We have got the best model in the previous step, here we need to extract his feaure and other information
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import os

checkpoint_path = 'my_checkpoints/best_checkpoint'
checkpoint = 'pre_trained_model'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=4, trust_remote_code=True)
model.eval() 

test_data_path = 'data/processed_test_dataset'
test_data = torch.load('../data/test.pt')
sequences = test_data['sequences']  
labels = test_data['labels']      

test_dataset = load_from_disk(test_data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

results = {
    'sequences': sequences,
    'labels': labels,
    'feature': [],
    'confidence': [],
    'prediction': []
}

with torch.no_grad():
    for i, batch in enumerate(test_dataset):
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items() if k in ['input_ids']}
        
        outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]  # Take the hidden state of the last layer
        last_hidden_state = hidden_states[:, -1, :]  # Gets the characteristics of the last token in each sequence
        results['feature'].append(last_hidden_state.cpu().numpy())
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=-1) 
        results['confidence'].append(confidence.item())
        results['prediction'].append(prediction.item())

output_path = 'hy_e2_step3_result.pt'
torch.save(results, output_path)
print(f"Results saved to {output_path}")