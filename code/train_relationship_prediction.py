import sys
import torch
from transformers import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from layers import RelationshipPredictor
from collections import Counter

parser = argparse.ArgumentParser(description='read arguments for training relationship prediction in MPDD')

parser.add_argument('--dir_path', nargs='?', type=str, default='/homes/rpujari/scratch1_fortytwo/DARPA/', help='Where everything should be saved')
parser.add_argument('--train_model', action='store_true', help="True if you want to train the model, else it will only evaluate.")
parser.add_argument('--test_mode', action='store_true', help="True if you want to evaluate on test set.")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model")
parser.add_argument("--save_path", type=str, default='/homes/rpujari/scratch1_fortytwo/DARPA/saved_models/bert-base_relation_prediction.pt', help='path to the saved trained model parameters')
parser.add_argument("--plm_name", type=str, default='hfl/chinese-bert-wwm', help='name of the transformers PLM to use')
parser.add_argument("--num_emotions", type=int, default=7, help='number of emotions labels in the data')
parser.add_argument("--em_emb_dim", type=int, default=30, help='size of emotion embedding to be used in the model')
parser.add_argument("--sizes", type=list, default=[1596, 1000, 20], help='sizes of feed forward layers of the model')
parser.add_argument("--num_positions", type=int, default=4, help='number of position labels in the data')
parser.add_argument("--num_relations", type=int, default=25, help='number of relation labels in the data')
parser.add_argument("--num_fields", type=int, default=5, help='number of field labels in the data')
parser.add_argument('--use_cuda', action='store_true', help="true is you want to use GPU")
parser.add_argument("--cuda_device", type=int, default=0, help='GPU to be used for the model training')
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate for the model training")
parser.add_argument("--class_weights_f", type=list, default=[1.] * 5, help='loss fn weights for each field class, they will be computed from training data')
parser.add_argument("--class_weights_p", type=list, default=[1.] * 4, help='loss fn weights for each position class, they will be computed from training data')
parser.add_argument("--class_weights_r", type=list, default=[1.] * 25, help='loss fn weights for each relation class, they will be computed from training data')

args = parser.parse_args()

def get_data(given_dataset):
    '''helper function to process the pandas dataframe for the MPDD dataset'''

    utterances = []
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
    def __init__(self, ut, prev_ut, em, prev_em, fs, ps, rs, em_ls, r_ls, p_ls, f_ls):
        self.utterance_encs = ut
        self.prev_utterance_encs = prev_ut
        self.emotions = em
        self.prev_emotions = prev_em
        self.fields = fs
        self.positions = ps
        self.relationships = rs
        self.emotion_labels = em_ls
        self.relation_labels = r_ls
        self.positions_labels = p_ls
        self.field_labels = f_ls

    def __getitem__(self, idx):
        item = {'ut_' + key: torch.tensor(val[idx]) for key, val in self.utterance_encs.items()}
        item.update({'prev_ut_' + key: torch.tensor(val[idx]) for key, val in self.prev_utterance_encs.items()})
        item['emotions'] = torch.tensor(self.emotion_labels[self.emotions[idx]])
        item['prev_emotions'] = torch.tensor(self.emotion_labels[self.prev_emotions[idx]])
        item['relations'] = torch.tensor(self.relation_labels[self.relationships[idx]])
        item['positions'] = torch.tensor(self.positions_labels[self.positions[idx]])
        item['fields'] = torch.tensor(self.field_labels[self.fields[idx]])
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
f_counts = Counter(tr_f)
p_counts = Counter(tr_p)
r_counts = Counter(tr_r)

print('Train data class distribution counts')
print(f_counts)
print(p_counts)
print(r_counts)

de_ut, de_prev_ut, de_em, de_prev_em, de_f, de_p, de_r = get_data(dev_csv)
te_ut, te_prev_ut, te_em, te_prev_em, te_f, te_p, te_r = get_data(test_csv)

#load label to int mappings
reln_em = load_dict(args.dir_path + 'relationship_emotion-label_dict.npy')
reln_r = load_dict(args.dir_path + 'relationship_relation-label_dict.npy')
reln_p = load_dict(args.dir_path + 'relationship_position-label_dict.npy')
reln_f = load_dict(args.dir_path + 'relationship_field-label_dict.npy')
for field in f_counts:
    args.class_weights_f[reln_f[field]] = 1.0 / f_counts[field]
for position in p_counts:
    args.class_weights_p[reln_p[position]] = 1.0 / p_counts[position]
for relation in r_counts:
    args.class_weights_r[reln_r[relation]] = 1.0 / r_counts[relation]

args.class_weights_f = torch.tensor(args.class_weights_f)
args.class_weights_p = torch.tensor(args.class_weights_p)
args.class_weights_r = torch.tensor(args.class_weights_r)

# set up the tokenizer and tokenize the data
tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained('hfl/chinese-bert-wwm')

tr_ut_enc = tokenizer(tr_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)
tr_prev_ut_enc = tokenizer(tr_prev_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)

de_ut_enc = tokenizer(de_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)
de_prev_ut_enc = tokenizer(de_prev_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)

te_ut_enc = tokenizer(te_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)
te_prev_ut_enc = tokenizer(te_prev_ut, truncation=True, padding=True, return_attention_mask=True, pad_to_max_length=True, max_length=512)

# set up the data as a dataset
train_dataset = CustomDataset(tr_ut_enc, tr_prev_ut_enc, tr_em, tr_prev_em, tr_f, tr_p, tr_r, reln_em, reln_r, reln_p, reln_f)
dev_dataset = CustomDataset(de_ut_enc, de_prev_ut_enc, de_em, de_prev_em, de_f, de_p, de_r, reln_em, reln_r, reln_p, reln_f)
test_dataset = CustomDataset(te_ut_enc, te_prev_ut_enc, te_em, te_prev_em, te_f, te_p, te_r, reln_em, reln_r, reln_p, reln_f)

# set up the dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# set up the model
model = RelationshipPredictor(args)
if args.use_cuda and torch.cuda.is_available():
    with torch.cuda.device(args.cuda_device):
        model.cuda()
        model.load_state_dict(torch.load(args.save_path))


if args.train_model:
    print("Training the model")
    sys.stdout.flush()
    model.train_model(train_loader, dev_loader, num_epochs=args.num_epochs, save_path=args.save_path)

if args.test_mode:

    model.load_state_dict(torch.load(args.save_path))
    dev_res = model.evaluate(dev_loader)
    print(dev_res[0])
    print(dev_res[1])
    print(dev_res[2])
    test_res = model.evaluate(test_loader)
    print(test_res[0])
    print(test_res[1])
    print(test_res[2])
