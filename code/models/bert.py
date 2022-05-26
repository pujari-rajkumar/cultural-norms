import torch
from torch import nn
from transformers import BertTokenizer, BertModel

class Bert(nn.Module):
    def __init__(self, bert_model_path="bert-base-chinese"):
        super().__init__()
        self.bert_model_path = bert_model_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        self.bert = BertModel.from_pretrained(self.bert_model_path)
        self.emb_size = self.bert.config.hidden_size

    def forward(self, seq_ids):
        """
        contextualize words using Bert obtain segment repr
        @param seq_ids (torch.tensor(b, Q)): Bert padded segment ids
        @return seg_reprs (torch.tensor(b, d))
        """
        attention_mask = (seq_ids != self.tokenizer.pad_token_id)

        bert_outs = self.bert(input_ids=seq_ids,
                                attention_mask=attention_mask)
        bert_last_out = bert_outs.last_hidden_state #(b, Q, d)
        
        seg_reprs = bert_last_out[:, 0, :] #(b, d) CLS repr
        return seg_reprs