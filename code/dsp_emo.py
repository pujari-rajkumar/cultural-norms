# different speaker prediction using emotion feat Bert
from copyreg import pickle
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Precision, Recall, F1Score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
import numpy as np
import json
from collections import OrderedDict
import random
import pickle

from dataset import MPDD
from models.bert import Bert
from models.bert import BertTokenizer
import utils

class EmoDSP(pl.LightningModule):
    def __init__(self, num_emotions):
        super().__init__()
        self.encoder = Bert()

        self.emb_size = self.encoder.emb_size
        self.hidden_size = 256
        self.dropout = 0.1

        self.emotion_emb_size = 48
        self.emotion_encoder = nn.Embedding(num_embeddings=num_emotions, embedding_dim=self.emotion_emb_size)

        self.classifier = nn.Sequential(
            OrderedDict([
          ('input', nn.Linear(self.emb_size + self.emotion_emb_size, self.hidden_size)),
          ('actvn1', nn.Tanh()),
          ('dropout1', nn.Dropout(self.dropout)),
          ('hidden', nn.Linear(self.hidden_size, self.hidden_size)),
          ('actvn2', nn.Tanh()),
          ('dropout2', nn.Dropout(self.dropout)),
          ('output', nn.Linear(self.hidden_size, 2))
        ])
        )

        self.pos_w = 0.4
        self.neg_w = 0.6

    def forward(self, x):
        sent_tensor, emotion_id_tensor = x
        with torch.no_grad():
            sent_embed = self.encoder(sent_tensor) #(b, d)
        emotion_embed = self.emotion_encoder(emotion_id_tensor) #(b, e)
        x_embed = torch.cat([sent_embed, emotion_embed], dim=-1) #(b, d + e)
        out = self.classifier(x_embed) #(b, 2)
        y_hat = torch.softmax(out, dim=-1)
        return y_hat

    def training_step(self, batch, batch_idx):
        self.encoder.eval()
        x, targets, _ = batch
        batch_size = len(targets)
        y_hat = self(x)
        log_y_hat = torch.log(y_hat)
        # create loss weight
        loss_w_tensor = torch.tensor([self.neg_w, self.pos_w], device=targets.device).repeat(batch_size, 1) #(b, 2)
        loss_w = torch.gather(loss_w_tensor, index=targets.unsqueeze(-1), dim=-1).squeeze(-1) #(b,)
        log_loss = torch.gather(log_y_hat, index=targets.unsqueeze(-1), dim=-1).squeeze(-1) * loss_w #(b,)
        loss = -torch.mean(log_loss)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, targets, _ = batch
        batch_size = len(targets)
        y_hat = self(x)
        log_y_hat = torch.log(y_hat)
        # create loss weight
        loss_w_tensor = torch.tensor([self.neg_w, self.pos_w], device=targets.device).repeat(batch_size, 1) #(b, 2)
        loss_w = torch.gather(loss_w_tensor, index=targets.unsqueeze(-1), dim=-1).squeeze(-1) #(b,)
        log_loss = torch.gather(log_y_hat, index=targets.unsqueeze(-1), dim=-1).squeeze(-1) * loss_w #(b,)
        loss = -torch.mean(log_loss)
        # Logging to TensorBoard by default
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        y_hat = self(x)
        return torch.argmax(y_hat, dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)
        return optimizer

# tokenizer
bert_tokenizer : BertTokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
#SEP
sep_token = bert_tokenizer.sep_token

def collate_fn(data):
    dialog_ids = [data_['dialog_id'] for data_ in data]
    utter_ids = [data_['utter_id'] for data_ in data]
    sents = [data_['sent'] for data_ in data]
    emotion_ids = [data_['emotion_id'] for data_ in data]
    labels = [data_['label'] for data_ in data]
    # tensorize labels
    # labels_list = []
    # for label in labels:
    #     y = np.zeros(2, dtype=int)
    #     y[label] = 1
    #     labels_list.append(y)
    # labels_batch = np.stack(labels_list, axis=0) #(b,)
    # y_batch = torch.from_numpy(labels_batch) #(b,)
    targets = torch.tensor(labels, dtype=torch.int64)
    # tensorize sents
    seqs_padded = utils.sents_to_seqs(sents, tokenizer=bert_tokenizer)
    seqs_ids = [bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in seqs_padded]
    sent_batch = torch.tensor(seqs_ids, dtype=torch.long) #(b, Q)
    # tensorize ids
    dialog_id_batch = torch.tensor(dialog_ids, dtype=torch.long)
    utter_id_batch = torch.tensor(utter_ids, dtype=torch.long)
    emotion_id_batch = torch.tensor(emotion_ids, dtype=torch.int64)
    return (sent_batch, emotion_id_batch), targets, (dialog_id_batch, utter_id_batch)

def run():

    # load Data
    data_path = "dir_path"
    dialogs_path = "{}/{}".format(data_path, "dialogue.json")
    dialogs = json.load(open(dialogs_path, "r"))

    # split train/dev/test
    dialog_keys = [k for k in dialogs.keys()]
    random.shuffle(dialog_keys)
    train_split = int(0.8 * len(dialogs))
    val_split = test_split = int(0.1 * len(dialogs))

    start = 0
    train_dialog_keys = dialog_keys[start : start + train_split]
    start += train_split
    val_dialog_keys = dialog_keys[start : start + val_split]
    start += val_split
    test_dialog_keys = dialog_keys[start : ]

    # create dataset
    train_dialogs = {k : dialogs[k] for k in train_dialog_keys}
    train_dataset = MPDD(train_dialogs, context=True, sep_token=sep_token)

    val_dialogs = {k : dialogs[k] for k in val_dialog_keys}
    val_dataset = MPDD(val_dialogs, context=True, sep_token=sep_token)

    test_dialogs = {k : dialogs[k] for k in test_dialog_keys}
    test_dataset = MPDD(test_dialogs, context=True, sep_token=sep_token)

    # tensorize data
    train_loader = DataLoader(train_dataset,
                            batch_size=8,
                            shuffle=True,
                            collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate_fn)

    # model
    num_emotions = train_dataset.num_emotions
    model = EmoDSP(num_emotions=num_emotions)
    for param in model.encoder.parameters():
            param.requires_grad = False

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath="model_path", #model save dir 
                                        filename="dsp_emo_feat_ctxt",
                                        save_weights_only=True,
                                        save_top_k=1, mode="min", monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    # loggings
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="log_dir") #log dir

    # training
    trainer = pl.Trainer(max_epochs=100, logger=tb_logger, accelerator="gpu", gpus=1, precision=16, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, train_loader, val_loader)
    
	# eval
    golds = []
    preds = []
    for batch_idx, batch in enumerate(test_loader):
        x, targets, _ = batch
        y_pred = model.predict_step(batch, batch_idx)
        batch_size = len(targets)
        for i in range(batch_size):
            target = targets[i].item()
            pred = y_pred[i].item()
            golds.append(target)
            preds.append(pred)

    # tensorize results
    golds_t = torch.tensor(golds, dtype=torch.long)
    preds_t = torch.tensor(preds, dtype=torch.long)

    # eval
    pre_func = Precision(num_classes=2, average=None)
    rec_func = Recall(num_classes=2, average=None)
    f1_func = F1Score(num_classes=2, average=None)

    P = pre_func(preds_t, golds_t)
    R = rec_func(preds_t, golds_t)
    F1 = f1_func(preds_t, golds_t)
    print("===speaker change===")
    print("pos P: %.3f, pos R: %.3f, pos F1: %.3f" %(P[1].item(), R[1].item(), F1[1].item()))
    print("neg P: %.3f, neg R: %.3f, neg F1: %.3f" %(P[0].item(), R[0].item(), F1[0].item()))

def pred():
     # load Data
    data_path = "dir_path"
    dialogs_path = "{}/{}".format(data_path, "dialogue.json")
    dialogs = json.load(open(dialogs_path, "r"))

    # split train/dev/test
    dialog_keys = [k for k in dialogs.keys()]
    random.shuffle(dialog_keys)
    train_split = int(0.8 * len(dialogs))
    val_split = test_split = int(0.1 * len(dialogs))

    start = train_split + val_split
    test_dialog_keys = dialog_keys[start : ]

    test_dialogs = {k : dialogs[k] for k in test_dialog_keys}
    test_dataset = MPDD(test_dialogs, context=True, sep_token=sep_token)

    test_loader = DataLoader(test_dataset,
                            batch_size=8,
                            shuffle=False,
                            collate_fn=collate_fn)

    # load best model
    model = EmoDSP.load_from_checkpoint("model_path/dsp.ckpt")
    model.eval()
    # save fn
    fns = {}
    # save test data
    test_data = {}
    for batch_idx, batch in enumerate(test_loader):
        x, targets, (dialog_id_batch, utter_id_batch) = batch
        y_pred = model.predict_step(batch, batch_idx)
        batch_size = len(targets)
        for i in range(batch_size):
            y_pred_i = y_pred[i].item()
            target_i = targets[i].item()
            dialog_key = str(dialog_id_batch[i].item())
            utter_num = utter_id_batch[i].item()
            if dialog_key not in test_data:
                test_data[dialog_key] = []
            test_data[dialog_key].append(utter_num)
            if y_pred_i != target_i and target_i == 1:
                if dialog_key not in fns:
                    fns[dialog_key] = []
                fns[dialog_key].append(utter_num)
    pickle.dump(fns, open("fns.pkl", "wb"))
    pickle.dump(test_data, open("test_data.pkl", "wb"))


if __name__ == "__main__":
    random.seed(0)
    run()
    # pred()
