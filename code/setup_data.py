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
        for i, utterance_dict in enumerate(all_utterances):
            if i > 0:
                prev_turn = all_utterances[i - 1]
                listeners = [item['name'] for item in utterance_dict['listener']]
                if prev_turn['speaker'] in listeners:
                    lidx = listeners.index(prev_turn['speaker'])
                    relation = utterance_dict['listener'][lidx]['relation']
                else:
                    relation = 'None'
                previous_utterance = prev_turn['utterance']
                
                return_utterance_emotion.append((utterance_dict['utterance'], previous_utterance, relation, utterance_dict['emotion']))
    return return_utterance_emotion

def get_relationship_prediction_tuples(all_data, data_split_keys):
    ret_data = []
    metadata = json.load(open(dir_path + 'mpdd/metadata.json'))
    rev_field = {'unknown': 'unknown'}
    rev_position = {'unknown': 'unknown'}

    for key in metadata['field']:
        for reln in metadata['field'][key]:
            if reln != 'unknown':
                rev_field[reln] = key

    for key in metadata['position']:
        for reln in metadata['position'][key]:
            if reln != 'unknown':
                rev_position[reln] = key

    for key in data_split_keys:
        all_utterances = all_data[key]
        for i, turn in enumerate(all_utterances):
            if i > 0:
                prev_turn = all_utterances[i - 1]
                listeners = [item['name'] for item in turn['listener']]
                if prev_turn['speaker'] in listeners:
                    lidx = listeners.index(prev_turn['speaker'])
                    relation = turn['listener'][lidx]['relation']
                    ret_data.append((turn['utterance'], prev_turn['utterance'], turn['emotion'], prev_turn['emotion'], rev_field[relation], rev_position[relation], relation))
    return ret_data

def create_dict_for_pandas(given_utterances, label_int_dicts, header, keep_text_labels=True, label_idx=[-1]):
    out_df_list = []

    for df_counter, given_utterance in enumerate(given_utterances):
        sample_dict = {'id': df_counter}

        for key, item in zip(header, given_utterance):
            sample_dict[key] = item


        if not keep_text_labels:
            for id_,li_dict in zip(label_idx, label_int_dicts):
                if type(id_) == int:
                    sample_dict[header[id_]] = li_dict[given_utterance[id_]]
                else:
                    print()
                    for id1 in id_:
                        sample_dict[header[id1]] = li_dict[given_utterance[id1]]
        out_df_list.append(sample_dict)
    out_df = pd.DataFrame(out_df_list)
    return out_df 

def create_data_tsvs(data_tuples, dir_path, header, label_idx=[-1], prefix='', keep_text_labels=True):
    train_set, dev_set, test_set = data_tuples


    # get the labels so we can convert them to int later
    label_int_dicts = []
    for id_ in label_idx:
        if type(id_) == tuple:
            label_counter = 0
            label_dict = {}
            for id1 in id_:
                for tup in train_set:
                    if tup[id1] not in label_dict:
                        label_dict[tup[id1]] = label_counter
                        label_counter += 1
                for tup in dev_set:
                    if tup[id1] not in label_dict:
                        label_dict[tup[id1]] = label_counter
                        label_counter += 1
                for tup in test_set:
                    if tup[id1] not in label_dict:
                        label_dict[tup[id1]] = label_counter
                        label_counter += 1
            np.save(dir_path + prefix + header[id_[0]] + '-label_dict.npy', np.asarray(label_dict))
            label_int_dicts.append(label_dict)
        else:
            label_counter = 0
            label_dict = {}
            for tup in train_set:
                if tup[id_] not in label_dict:
                    label_dict[tup[id_]] = label_counter
                    label_counter += 1
            for tup in dev_set:
                if tup[id_] not in label_dict:
                    label_dict[tup[id_]] = label_counter
                    label_counter += 1
            for tup in test_set:
                if tup[id_] not in label_dict:
                    label_dict[tup[id_]] = label_counter
                    label_counter += 1
            np.save(dir_path + prefix + header[id_] + '-label_dict.npy', np.asarray(label_dict))
            label_int_dicts.append(label_dict)

    # set the pandas dataframes
    train_df = create_dict_for_pandas(train_set, label_int_dicts, header, keep_text_labels=keep_text_labels, label_idx=label_idx)
    dev_df = create_dict_for_pandas(dev_set, label_int_dicts, header, keep_text_labels=keep_text_labels, label_idx=label_idx)
    test_df = create_dict_for_pandas(test_set, label_int_dicts, header, keep_text_labels=keep_text_labels, label_idx=label_idx)

    # print(dev_df.head())

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


    ### Emotion Detection data setup ###
    #get tuples of relationship prediction from data
    emotion_train_set = get_utterance_emotion_tuples(dialogue_data, train_set)
    emotion_dev_set = get_utterance_emotion_tuples(dialogue_data, dev_set)
    emotion_test_set = get_utterance_emotion_tuples(dialogue_data, test_set)

    emotion_data_tuples = (emotion_train_set, emotion_dev_set, emotion_test_set)


    emotion_header = ['text', 'prev_utterance', 'relation', 'label']
    create_data_tsvs(emotion_data_tuples, dir_path, emotion_header, prefix='emotions_', keep_text_labels=False)



    #print(test_set)

    ### Relationship Prediction Dataset ###
    #get tuples of relationship prediction from data
    reln_train_set = get_relationship_prediction_tuples(dialogue_data, train_set)
    reln_dev_set = get_relationship_prediction_tuples(dialogue_data, dev_set)
    reln_test_set = get_relationship_prediction_tuples(dialogue_data, test_set)

    reln_data_tuples = (reln_train_set, reln_dev_set, reln_test_set)

    reln_header = ['utterance', 'prev_utterance', 'emotion', 'prev_emotion', 'field', 'position', 'relation']
    label_idx = [(2, 3), 4, 5, 6]
    create_data_tsvs(reln_data_tuples, dir_path, reln_header, label_idx=label_idx, prefix='relationship_', keep_text_labels=True)
