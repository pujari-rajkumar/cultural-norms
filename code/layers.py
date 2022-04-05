import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import *

class RelationshipPredictor(nn.Module):

    def __init__(self, args):
        self.plm_model = args.plm_class.from_pretrained(args.plm_name)
        self.emotion_embedding = nn.Embedding(args.num_emotions, args.em_emb_dim)
        self.layers = []
        i = 0
        for s1, s2 in zip(args.sizes[:-1], args.sizes[1:]):
            self.layers.append(nn.Linear(s1, s2))
            self.register_parameter('weight-layer-' + str(i), self.layers[-1].weight)
            self.register_parameter('bias-layer-' + str(i), self.layers[-1].bias)
            nn.init.xavier_uniform_(self.layers[-1].weight)
            i += 1
        self.nl = nn.Tanh()
        self.position_layer = nn.Linear(args.sizes[-1], args.num_positions)
        self.rleation_layer = nn.Linear(args.sizes[-1], args.num_relations)
        self.field_layer = nn.Linear(args.sizes[-1], args.num_fields)

    def forward(self, ut, prev_ut, em, prev_em):
        ut_emb = self.plm_model(ut['input_ids'].to(device), attention_mask=ut['attention_mask'].to(device))[1]
        prev_ut_emb = self.plm_model(prev_ut['input_ids'].to(device), attention_mask=prev_ut['attention_mask'].to(device))[1]
        em_emb = self.emotion_embedding(em)
        prev_em_emb = self.emotion_embedding(prev_em)
        x = torch.cat([ut_emb, prev_ut_emb, em_emb, prev_em_emb], dim=1)
        for layer in self.layers:
            x = self.nl(layer(x))
        position_out = F.softmax(self.position_layer(x), dim=1)
        relation_out = F.softmax(self.relation_out(x), dim=1)
        field_out = F.softmax(self.field_layer(x), dim=1)

        return field_out, position_out, relation_out

