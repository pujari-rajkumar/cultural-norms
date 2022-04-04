import numpy as np
import json
import sklearn
import sklearn.model_selection
import pandas as pd
import argparse


def get_utterance_emotion_tuples(all_data, data_split_keys): 
    return_utterance_emotion = []
    for key in data_split_keys:
        all_utterances = all_data[key]
        for utterance_dict in all_utterances:
            return_utterance_emotion.append((utterance_dict['utterance'], utterance_dict['emotion']))
    return return_utterance_emotion

def get_relationship_prediction_tuples(all_data, data_split_keys):
    ret_data = []
    for key in data_split_keys:
        all_utterances = all_data[key]
        for i, turn in enumerate(all_utterances):
            if i > 0:
                prev_turn = all_utterances[i - 1]
                listeners = [item['name'] for item in turn['listener']]
                if prev_turn['speaker'] in listeners:
                    lidx = listeners.index(prev_turn['speaker'])
                    ret_data.append((turn['utterance'], prev_turn['utterance'], turn['emotion'], prev_turn['emotion'], turn['listener'][lidx]['relation']))
    return ret_data

def create_dict_for_pandas(given_utterances, label_int_dict, header):
    out_df_list = []
    for df_counter, given_utterance in enumerate(given_utterances):
        sample_dict = {'id': df_counter}
        for key, item in zip(header, given_utterance):
            sample_dict[key] = item
        out_df_list.append(sample_dict)
    #for tup in out_df_list:
        #print(tup)
        #print('-------------------------\n')

    out_df = pd.DataFrame(out_df_list)
    return out_df 

def create_data_tsvs(data_tuples, dir_path, header, prefix=''):
    train_set, dev_set, test_set = data_tuples

    # get the labels so we can convert them to int later
    label_counter = 1
    label_dict = {}
    for tup in train_set:
        if tup[-1] not in label_dict:
            #print(tup[-1])
            label_dict[tup[-1]] = label_counter
            label_counter += 1
    np.save(dir_path + prefix + 'label_dict.npy', np.asarray(label_dict))

    # set the pandas dataframes
    train_df = create_dict_for_pandas(train_set, label_dict, header)
    dev_df = create_dict_for_pandas(dev_set, label_dict, header)
    test_df = create_dict_for_pandas(test_set, label_dict, header)

    # save the pandas dataframes and we will load these later in train_model.py
    train_df.to_csv(dir_path + prefix + 'train.tsv', sep='\t', index=False, header=True, columns=train_df.columns)
    dev_df.to_csv(dir_path + prefix + 'dev.tsv', sep='\t', index=False, header=True, columns=dev_df.columns)
    test_df.to_csv(dir_path + prefix + 'test.tsv', sep='\t', index=False, header=True, columns=test_df.columns)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up data for prediction tasks')
    parser.add_argument('--dir_path', nargs='?', type=str, default='/u/zaphod_s3/mehta52/DARPA_project/', help='Where everything should be saved')

    args = parser.parse_args()

    dir_path = args.dir_path 
    
    # load the data
    f = open(dir_path + '/mpdd/dialogue.json')
    dialogue_data = json.load(f)

    # split the keys into a training/dev/test set
    train_dev_set, test_set = sklearn.model_selection.train_test_split(list(dialogue_data.keys()), test_size=0.20, random_state=4056)
    train_set, dev_set = sklearn.model_selection.train_test_split(train_dev_set, test_size=0.20, random_state=4056)

    #print(train_set)
    #print(dev_set)
    #print(test_set)

    ### Emotion Detection data setup ###
    #get tuples of relationship prediction from data
    emotion_train_set = get_utterance_emotion_tuples(dialogue_data, train_set)
    emotion_dev_set = get_utterance_emotion_tuples(dialogue_data, dev_set)
    emotion_test_set = get_utterance_emotion_tuples(dialogue_data, test_set)

    emotion_data_tuples = (emotion_train_set, emotion_dev_set, emotion_test_set)

    emotion_header = ['text', 'label']
    create_data_tsvs(emotion_data_tuples, dir_path, emotion_header, 'emotions_')

    ### Relationship Prediction Dataset ###
    #get tuples of relationship prediction from data
    reln_train_set = get_relationship_prediction_tuples(dialogue_data, train_set)
    reln_dev_set = get_relationship_prediction_tuples(dialogue_data, dev_set)
    reln_test_set = get_relationship_prediction_tuples(dialogue_data, test_set)

    reln_data_tuples = (reln_train_set, reln_dev_set, reln_test_set)

    reln_header = ['utterance', 'prev_utteracne', 'emotion', 'prev_emotion', 'relation']
    create_data_tsvs(reln_data_tuples, dir_path, reln_header, 'relationship_')
