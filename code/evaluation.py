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
from argostranslate import package, translate



# eval functions
from sklearn.metrics import f1_score
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')
    
def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}')
        print("Accuracy Percentage : " +  str(len(y_preds[y_preds==label])/len(y_true)))
        print("")


# preds_flat = np.argmax(val_predictions, axis=1).flatten()
#     labels_flat = val_true_values.flatten()

#     print(classification_report(y_pred=preds_flat, y_true=labels_flat))

#     print(label_dict_reverse)

#     matrix = confusion_matrix(y_true=labels_flat, y_pred=preds_flat)
#     print(matrix.diagonal()/matrix.sum(axis=1))

def evaluate(model, dataloader_val, device, label_dict=None, args=None):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []

    wrong_text_to_prediction_mapping = {}
    correct_text_to_prediction_mapping = {}
    top_wrong_relation = []
    
    for batch in tqdm(dataloader_val):

        original_texts = batch['original_text']

        prev_utterance_texts = batch['prev_utterance']

        relations = batch['relation']
                
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():        
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

        preds_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = labels.flatten()

        if args.do_analysis_on_mistakes:
            if len(wrong_text_to_prediction_mapping) > 100:
                break

            for counter, (x, y) in enumerate(zip(preds_flat, labels_flat)):
                if x != y:
                    wrong_text_to_prediction_mapping[original_texts[counter]] = (x, y, prev_utterance_texts[counter], relations[counter])
                    top_wrong_relation.append(relations[counter])
                else:
                    correct_text_to_prediction_mapping[original_texts[counter]] = (x, y, prev_utterance_texts[counter], relations[counter])


    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    if args.do_analysis_on_mistakes:
        np.save(args.dir_path +  '/wrong_text_to_prediction_mapping.npy', np.asarray(dict(wrong_text_to_prediction_mapping)))
        np.save(args.dir_path +  '/correct_text_to_prediction_mapping.npy', np.asarray(dict(correct_text_to_prediction_mapping)))

    installed_languages = translate.get_installed_languages()
    translation_zh_en = installed_languages[1].get_translation(installed_languages[0])



    if args.do_analysis_on_mistakes and label_dict is not None:
        print("Results for analysis:")
        label_dict_inverse = {v: k for k, v in label_dict.items()}

        print(label_dict_inverse)

        for given_text_wrong in wrong_text_to_prediction_mapping:
            # translate text
            print(given_text_wrong)
            print(translation_zh_en.translate(given_text_wrong))
            print("Prev utterance: " + translation_zh_en.translate(wrong_text_to_prediction_mapping[given_text_wrong][2]))
            print("Relation: " + wrong_text_to_prediction_mapping[given_text_wrong][3])
            try:
                print("Predicted Label " + str(label_dict_inverse[wrong_text_to_prediction_mapping[given_text_wrong][0].item()]))
                print("True Label " + str(label_dict_inverse[wrong_text_to_prediction_mapping[given_text_wrong][1].item()]))
                
            except:
                continue
            print("")

        print(Counter(top_wrong_relation))


            
    return loss_val_avg, predictions, true_vals



