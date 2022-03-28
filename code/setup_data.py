import numpy as np
import json
import sklearn
import sklearn.model_selection
import pandas as pd

dir_path = '/u/zaphod_s3/mehta52/DARPA_project/'

def get_utterance_emotion_tuples(all_data, data_split_keys):
    
    return_utterance_emotion = []
    for x in data_split_keys:
        all_utterances = all_data[x]
        for utterance_dict in all_utterances:
            return_utterance_emotion.append((utterance_dict['utterance'], utterance_dict['emotion']))
            
    return return_utterance_emotion

def create_dict_for_pandas(given_utterances, label_int_dict):
    out_df_list = []
    for df_counter, given_utterance in enumerate(given_utterances):
        out_df_list.append({'id':df_counter, 'label': label_int_dict[given_utterance[1]], 'alpha':['a']*100, 'text': given_utterance[0]})
                   
    out_df = pd.DataFrame(out_df_list)
            
    return out_df 

# load the data
f = open(dir_path + '/mpdd/dialogue.json')
dialogue_data = json.load(f)


# split the keys into a training/dev/test set
train_dev_set, test_set = sklearn.model_selection.train_test_split(list(dialogue_data.keys()), test_size=20)
train_set, dev_set = sklearn.model_selection.train_test_split(train_dev_set, test_size=20)

# get tuples of utterance and emotion from the dataset
train_set_utterances = get_utterance_emotion_tuples(dialogue_data, train_set)
dev_set_utterances = get_utterance_emotion_tuples(dialogue_data, dev_set)
test_set_utterances = get_utterance_emotion_tuples(dialogue_data, test_set)

# get the labels so we can convert them to int later
label_counter = 1
label_dict = {}
for x in train_set_utterances:
    if x[1] not in label_dict:
        label_dict[x[1]] = label_counter
        label_counter += 1

# set the pandas dataframes
train_df = create_dict_for_pandas(train_set_utterances, label_dict)
dev_df = create_dict_for_pandas(dev_set_utterances, label_dict)
test_df = create_dict_for_pandas(test_set_utterances, label_dict)

# save the pandas dataframes and we will load these later in train_model.py
train_df.to_csv(dir_path + 'train_emotion.tsv', sep='\t', index=False, header=True, columns=train_df.columns)
train_df.to_csv(dir_path + 'dev_emotions.tsv', sep='\t', index=False, header=True, columns=dev_df.columns)
train_df.to_csv(dir_path + 'test_emotions.tsv', sep='\t', index=False, header=True, columns=test_df.columns)

