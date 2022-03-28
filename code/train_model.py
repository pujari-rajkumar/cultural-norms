import pickle
import os
import sys
from datetime import datetime
import threading
import math
import json
import torch
from transformers import *
import re
from nltk.corpus import stopwords
import urllib
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from evaluation import *
import argparse


parser = argparse.ArgumentParser(description='download twitter data')

parser.add_argument('--dir_path', nargs='?', type=str, default='/u/zaphod_s3/mehta52/DARPA_project/', help='Where everything should be saved')
parser.add_argument('--train_model', action='store_true', help="True if you want to train the model, else it will evaluate.")
parser.add_argument("--epoch_to_load", type=int, default=0, help="What epoch trained model you want to load")
parser.add_argument("--num_epochs", type=int, default=1, help="How many epochs you want to run")

args = parser.parse_args()

def get_text_labels(given_dataset):
    '''helper function to process the pandas dataframe for the MPDD dataset'''

    return_text = []
    return_labels = []
    
    for index, row in given_dataset.iterrows():
        return_text.append(row['text'])
        return_labels.append(row['label'])
        
    return return_text, return_labels


class CustomDataset(torch.utils.data.Dataset):
    '''class for our customDataset that we can pass into PyTorch via dataloader'''
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# load the data that we set up earlier
train_csv = pd.read_csv(args.dir_path + 'train_emotion.tsv', sep='\t')
dev_csv = pd.read_csv(args.dir_path + 'dev_emotions.tsv', sep='\t')
test_csv = pd.read_csv(args.dir_path + 'test_emotions.tsv', sep='\t')
# process the data 
train_texts, train_labels = get_text_labels(train_csv)
dev_texts, dev_labels = get_text_labels(train_csv)
test_texts, test_labels = get_text_labels(train_csv)
print(Counter(train_labels))

# set up the tokenizer and tokenize the data
tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained('hfl/chinese-bert-wwm')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True)
val_encodings = tokenizer(dev_texts, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True)

# set up the data as a dataset
train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, dev_labels)
test_dataset = CustomDataset(test_encodings, test_labels)
# set up the dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# set up the model
model_class = BertForSequenceClassification
config_class = BertConfig
model = model_class.from_pretrained('hfl/chinese-bert-wwm', num_labels=len(Counter(train_labels)) + 1)
device = torch.device("cuda")
model.to(device)

print("Training the model")
sys.stdout.flush()

optim = AdamW(model.parameters(), lr=5e-5)

if args.train_model:
    for epoch in tqdm(range(args.num_epochs)):

        model.train()

        loss_train_total = 0
    
        progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
            optim.step()

        torch.save(model.state_dict(), f'/u/zaphod_s3/mehta52/DARPA_project/saved_models/finetuned_BERT_epoch_{epoch}.model')

        loss_train_avg = loss_train_total/len(train_loader)            
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(model, val_loader, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

else:

    model.load_state_dict(torch.load(args.dir_path + '/saved_models/finetuned_BERT_epoch_' + str(args.epoch_to_load) + '.model', map_location=torch.device('cpu')))

    model.eval()
    
    # evaluate the model and save the predictions so we can further analyze it later
    print("Evaluating the model")
    sys.stdout.flush()
    val_loss, predictions, true_vals = evaluate(model, val_loader, device)
    np.save(args.dir_path + '/results_emotion/val_predictions.npy', np.asarray(predictions))
    np.save(args.dir_path + '/results_emotion/val_true_values.npy', np.asarray(true_vals))
    val_f1 = f1_score_func(predictions, true_vals)

    test_loss, test_predictions, test_vals = evaluate(model, test_loader, device)
    np.save(args.dir_path + '/results_emotion/test_predictions.npy', np.asarray(test_predictions))
    np.save(args.dir_path +  + '/results_emotion/test_true_values.npy', np.asarray(test_vals))
    val_f1 = f1_score_func(predictions, test_vals)


