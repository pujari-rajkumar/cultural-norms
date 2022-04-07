from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import *

class RelationshipPredictor(nn.Module):
    
    def __init__(self, args):
        super(RelationshipPredictor, self).__init__()
        self.plm_model = BertModel.from_pretrained(args.plm_name)
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
        self.relation_layer = nn.Linear(args.sizes[-1], args.num_relations)
        self.field_layer = nn.Linear(args.sizes[-1], args.num_fields)
        self.f_loss_fn = nn.CrossEntropyLoss(weight=args.class_weights_f)
        self.p_loss_fn = nn.CrossEntropyLoss(weight=args.class_weights_p)
        self.r_loss_fn = nn.CrossEntropyLoss(weight=args.class_weights_r)
        self.optimizer = AdamW(self.parameters(), lr=args.lr)
        self.args = args
    
    def forward(self, batch):
        with torch.no_grad():
            ut_emb = self.plm_model(batch['ut_input_ids'], attention_mask=batch['ut_attention_mask'])[1]
            prev_ut_emb = self.plm_model(batch['prev_ut_input_ids'], attention_mask=batch['prev_ut_attention_mask'])[1]
        em_emb = self.emotion_embedding(batch['emotions'])
        prev_em_emb = self.emotion_embedding(batch['prev_emotions'])
        x = torch.cat([ut_emb, prev_ut_emb, em_emb, prev_em_emb], dim=1)
        for layer in self.layers:
            x = self.nl(layer(x))
        position_out = F.softmax(self.position_layer(x), dim=1)
        relation_out = F.softmax(self.relation_layer(x), dim=1)
        field_out = F.softmax(self.field_layer(x), dim=1)
        
        return field_out, position_out, relation_out
    
    def get_results(self, batch_preds, batch_ys, batch_outs, loss_fn):
        pred_y = torch.cat(batch_preds, dim=0)
        data_y = torch.cat(batch_ys, dim=0)
        pred_out = torch.cat(batch_outs, dim=0)
        acc = sum((pred_y == data_y).float()) / data_y.size(0)
        f1_mi = f1_score(data_y.cpu().data, pred_y.cpu().data, average='micro')
        f1_ma = f1_score(data_y.cpu().data, pred_y.cpu().data, average='macro')
        con_mat = confusion_matrix(data_y.cpu().data, pred_y.cpu().data)
        val_loss = loss_fn(pred_out, data_y)
        res = acc.cpu(), val_loss.cpu(), (f1_mi, f1_ma, con_mat)
        return res
    
    def evaluate(self, data_loader):
        self.eval()
        batch_fouts = []
        batch_fpreds = []
        batch_fys = []
        batch_pouts = []
        batch_ppreds = []
        batch_pys = []
        batch_routs = []
        batch_rpreds = []
        batch_rys = []
        for batch in tqdm(data_loader):
            if self.args.use_cuda and torch.cuda.is_available():
                with torch.cuda.device(self.args.cuda_device):
                    batch = {key: val.cuda() for key, val in batch.items()}
            batch_fout, batch_pout, batch_rout = self.forward(batch)
            batch_fpred = torch.argmax(batch_fout, dim=1)
            batch_ppred = torch.argmax(batch_pout, dim=1)
            batch_rpred = torch.argmax(batch_rout, dim=1)
            batch_fouts.append(batch_fout)
            batch_fpreds.append(batch_fpred)
            batch_fys.append(batch['fields'])
            batch_pouts.append(batch_pout)
            batch_ppreds.append(batch_ppred)
            batch_pys.append(batch['positions'])
            batch_routs.append(batch_rout)
            batch_rpreds.append(batch_rpred)
            batch_rys.append(batch['relations'])
        res_f = self.get_results(batch_fpreds, batch_fys, batch_fouts, self.f_loss_fn)
        res_p = self.get_results(batch_ppreds, batch_pys, batch_pouts, self.p_loss_fn)
        res_r = self.get_results(batch_rpreds, batch_rys, batch_routs, self.r_loss_fn)
        return (res_f, res_p, res_r)
        
    def train_model(self, train_loader, dev_loader, num_epochs=10, save_path='/homes/rpujari/scratch1_fortytwo/DARPA/saved_models/relation_model.pkl'):
        max_val = -1
        for epoch in tqdm(range(num_epochs)):
            self.train()
            loss_train_total = 0
            progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
            for batch in progress_bar:
                if self.args.use_cuda and torch.cuda.is_available():
                    with torch.cuda.device(self.args.cuda_device):
                        batch = {key: val.cuda() for key, val in batch.items()}
                labels = batch['relations']
                field_out, position_out, relation_out = self.forward(batch)
                reln_loss = self.r_loss_fn(relation_out, batch['relations'])
                field_loss = self.f_loss_fn(field_out, batch['fields'])
                pos_loss = self.p_loss_fn(position_out, batch['positions'])
                loss = reln_loss + field_loss + pos_loss
                loss_train_total += loss.to('cpu').item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            loss_train_avg = loss_train_total/len(train_loader)
            tqdm.write(f'Training loss: {loss_train_avg}')
            sys.stdout.flush()
                
            dev_fres, dev_pres, dev_rres = self.evaluate(dev_loader)
            dev_acc, dev_loss, (f1_mi, f1_ma, cm) = dev_rres
            if f1_ma > max_val:
                max_val = f1_ma
                torch.save(self.state_dict(), save_path)   
