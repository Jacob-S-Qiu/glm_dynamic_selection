# Data initialization, since NTv2 requires the tokenize of the data input is more convenient, so it is handled in advance here

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments,Trainer
from datasets import load_dataset, Dataset
import torch
import os
import numpy as np

base_checkpoints_path = "pre_trained_model"

tokenizer = AutoTokenizer.from_pretrained(base_checkpoints_path,trust_remote_code=True)
max_length = tokenizer.model_max_length
model = AutoModelForSequenceClassification.from_pretrained(base_checkpoints_path,trust_remote_code=True,num_labels=2)
config = AutoConfig.from_pretrained(base_checkpoints_path,trust_remote_code=True)
print(config)


train_data = torch.load('data/train.pt')
valid_data = torch.load('data/valid.pt')
test_data = torch.load('data/test.pt')


ds_train= Dataset.from_dict({"data": train_data['sequences'],'labels':train_data['labels']})
ds_valid= Dataset.from_dict({"data": valid_data['sequences'],'labels':valid_data['labels']})
ds_test= Dataset.from_dict({"data": test_data['sequences'],'labels':test_data['labels']})


def tokenize_function(examples):
    outputs = tokenizer.batch_encode_plus(examples["data"],max_length=max_length,truncation=True)
    return outputs


tokenized_train = ds_train.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
tokenized_train.set_format("torch")
tokenized_train = {
    'input_ids': tokenized_train['input_ids'],
    'attention_mask': tokenized_train['attention_mask'],
    'labels': tokenized_train['labels'] 
}
torch.save(tokenized_train, 'data/tokenized_train.pt') 


tokenized_valid = ds_valid.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
tokenized_valid.set_format("torch")
tokenized_valid = {
    'input_ids': tokenized_valid['input_ids'],
    'attention_mask': tokenized_valid['attention_mask'],
    'labels': tokenized_valid['labels']  # 假设 labels 是在 train_data_5 中
}
torch.save(tokenized_valid, 'data/tokenized_valid.pt') 


tokenized_test = ds_test.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
tokenized_test.set_format("torch")
tokenized_test = {
    'input_ids': tokenized_test['input_ids'],
    'attention_mask': tokenized_test['attention_mask'],
    'labels': tokenized_test['labels']  # 假设 labels 是在 train_data_5 中
}
torch.save(tokenized_test, 'data/tokenized_test.pt') 





