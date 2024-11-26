# Get its feature data from the best model you've trained and found in the previous step

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import CDGPTSequencePrediction
from tokenizer import SentencePieceTokenizer
from config import get_config

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
        
        return sequence, torch.tensor(tokens), label  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = get_config()
cfg.model.num_classes = 4  
tokenizer_path = "checkpoints/tokenizer.model"
cfg.tokenizer.path = tokenizer_path
tokenizer = SentencePieceTokenizer(tokenizer_path)
cfg.tokenizer.pad_token_id = tokenizer.pad

model_path = 'my_checkpoints/best_checkpoint.pth'
model = CDGPTSequencePrediction(cfg)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.pad_token_id = tokenizer.pad
model = model.to(device)
model.eval()

features = []

def hook_fn(module, input, output):
    features.append(output.detach().cpu()) 

hook_handle = model.cls_head.dense.register_forward_hook(hook_fn)

test_file = "data/test.pt"
test_dataset = SequenceDataset(test_file, tokenizer, max_length=1024)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

results = {
    "sequences": [],
    "labels": [],
    "feature": [],
    "confidence": [],
    "prediction": []
}

with torch.no_grad():
    for original_sequence, tokens, label in test_loader:
        tokens = tokens.to(device)
        label = label.to(device)

        features.clear()

        outputs = model(tokens)
        
        logits = outputs["output"] 
        predictions = F.softmax(logits, dim=-1)
        max_confidence, pred_class = torch.max(predictions, dim=1)
        
        results["sequences"].append(original_sequence[0])  # Save the original text sequence
        results["labels"].append(label.item()) 
        results["feature"].append(features[0].squeeze(0))  # Remove the batch dimension using the feature captured by hook
        results["confidence"].append(max_confidence.item())  
        results["prediction"].append(pred_class.item())

hook_handle.remove()

output_path = "cd_e2_step2_result.pt"
torch.save(results, output_path)
print(f"Results saved to {output_path}")
















