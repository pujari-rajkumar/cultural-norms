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
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='read arguments for training relationship prediction in MPDD')

parser.add_argument('--dir_path', nargs='?', type=str, default='/u/zaphod_s3/mehta52/DARPA_project/', help='Where everything should be saved')
parser.add_argument('--train_model', action='store_true', help="True if you want to train the model, else it will only evaluate.")
parser.add_argument('--do_eval_and_save', action='store_true', help="True if you want to evaluate the model and save the results.")
parser.add_argument('--check_results', action='store_true', help="True if have already saved the results by running do_eval_and_save and now you want to run some analysis on them. For now we check the accuracy of each class for emotion classification.")
parser.add_argument("--epoch_to_load", type=int, default=0, help="What epoch trained model you want to load")
parser.add_argument("--num_epochs", type=int, default=1, help="How many epochs you want to run")

args = parser.parse_args()

def get_data(given_dataset):
    '''helper function to process the pandas dataframe for the MPDD dataset'''

    utteraces = []
    prev_utterances = []
    emotions = []
    prev_emotions = []
    fields = []
    positions = []
    relations = []

    for index, row in given_dataset.iterrows():
        utterances.append(row['utterance'])
        prev_utterances.append(row['prev_utterance'])
        emotions.append(row['emotion'])
        prev_emotions.append(row['prev_emotion'])
        fields.append(row['field'])
        positions.append(row['position'])
        relations.append(row['relation'])
        
    return utterances, prev_utterances, emotions, prev_emotions, fields, positions, relations


class CustomDataset(torch.utils.data.Dataset):
    '''class for our customDataset that we can pass into PyTorch via dataloader'''
    def __init__(self, ut, prev_ut, em, prev_em, fs, ps, rs, task='relationship'):
        self.utterance_encs = ut
        self.prev_utterance_encs = prev_ut
        self.emotions = em
        self.prev_emotions = prev_em
        self.fields = fs
        self.positions = ps
        self.relationships = rs
        self.task = task

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.task == 'relationship':
            item['labels'] = torch.tensor(self.relationships[idx])
        elif self.task = 'position':
            item['labels'] = torch.tensor(self.positions[idx])
        elif self.task == 'field':
            item['labels'] = torch.tensor(self.fields[idx])
        return item

    def __len__(self):
        return len(self.relationships)

def load_dict(dict_path):
    '''helper function to load a dictionary given a dict_path'''
    out_dict = defaultdict(list)
    old_out_dict = np.load(dict_path, allow_pickle=True)
    out_dict.update(old_out_dict.item())
    return out_dict


# load the data that we set up using setup_data.py
train_csv = pd.read_csv(args.dir_path + 'relationship_train.tsv', sep='\t')
dev_csv = pd.read_csv(args.dir_path + 'relationship_dev.tsv', sep='\t')
test_csv = pd.read_csv(args.dir_path + 'relationship_test.tsv', sep='\t')

# process the data 
tr_ut, tr_prev_ut, tr_em, tr_prev_em, tr_f, tr_p, tr_r = get_data(train_csv)
de_ut, de_prev_ut, de_em, de_prev_em, de_f, de_p, de_r = get_data(dev_csv)
te_ut, te_prev_ut, te_em, te_prev_em, te_f, te_p, te_r = get_data(test_csv)


# set up the tokenizer and tokenize the data
tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained('hfl/chinese-bert-wwm')

tr_ut_enc = tokenizer(tr_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)
tr_prev_ut_enc = tokenizer(tr_prev_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)

de_ut_enc = tokenizer(de_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)
de_prev_ut_enc = tokenizer(de_prev_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)

tr_ut_enc = tokenizer(te_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)
te_prev_ut_enc = tokenizer(te_prev_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)

# set up the data as a dataset
train_dataset = CustomDataset(tr_ut_enc, tr_prev_ut_enc, tr_em, tr_prev_em, tr_f, tr_p, tr_r)
dev_dataset = CustomDataset(de_ut_enc, de_prev_ut_enc, de_em, de_prev_em, de_f, de_p, de_r)
test_dataset = CustomDataset(te_ut_enc, te_prev_ut_enc, te_em, te_prev_em, te_f, te_p, te_r)

# set up the dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    best_val_f1 = 0.0
    label_dict = load_dict(args.dir_path + 'label_dict.npy')
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

        loss_train_avg = loss_train_total/len(train_loader)            
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(model, val_loader, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        accuracy_per_class(preds=predictions, labels=true_vals, label_dict=label_dict)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("Best validation F1 so far is " + str(best_val_f1))

        torch.save(model.state_dict(), f'/u/zaphod_s3/mehta52/DARPA_project/saved_models/finetuned_BERT_epoch_{epoch}.model')

elif args.do_eval_and_save:

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
    np.save(args.dir_path +  '/results_emotion/test_true_values.npy', np.asarray(test_vals))
    test_f1 = f1_score_func(preds=test_predictions, labels=test_vals)
    print("Test f1 is " + str(test_f1))

    label_dict = load_dict(args.dir_path + 'label_dict.npy')

    accuracy_per_class(preds=test_predictions, labels=test_vals, label_dict=label_dict)

elif args.check_results:

    # check the F1 score of each class
    val_predictions = np.load(args.dir_path + '/results_emotion/val_predictions.npy')
    val_true_values = np.load(args.dir_path + '/results_emotion/val_true_values.npy')

    # load the label dict so we can map the classes. This was computed in setup_data.py
    label_dict = load_dict(args.dir_path + 'label_dict.npy')

    accuracy_per_class(preds=val_predictions, labels=val_true_values, label_dict=label_dict)
