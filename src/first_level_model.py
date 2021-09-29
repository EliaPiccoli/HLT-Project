import copy
import math
import pickle
import torch
from tqdm import tqdm
from torch import nn
from torch._C import dtype
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from data_parser import Parser
from sklearn.model_selection import train_test_split
from dataset import BalancedCTTDataset
from object_transformer import Transformer
import json
import time

# set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CTTClassifier(nn.Module):
    def __init__(self, left_dims=(2e4, 128), left_inp_dim=128, seq_len=30, nhead=8, dim_feedforward=2048,
                 transformer_dropout=0.7, decoder_dropout=0.2, activation='relu', num_layers=2, dec_dims=(512,)):
        super().__init__()
        output_dim = 15 # number of classes
        self.left_model = Transformer(dims=left_dims, nhead=nhead, dim_feedforward=dim_feedforward, dropout=transformer_dropout,
                                      activation=activation, num_layers=num_layers)
        self.final_left = nn.Linear(seq_len * left_dims[1], left_inp_dim)
        office_onehot = 20
        # decoder with 1 single layer
        self.decoder = nn.Sequential(nn.Linear(left_inp_dim + office_onehot, output_dim))
        self.drop = nn.Dropout(decoder_dropout)

    def forward(self, obj, off):
        enc_obj = self.left_model(obj)
        enc_obj = torch.flatten(enc_obj, start_dim=1)
        left_enc = F.relu(self.final_left(enc_obj))
        classifier_input = torch.cat((left_enc, off), dim=1)
        x = self.drop(classifier_input)
        x = self.decoder(x)
        return x

def train(model: CTTClassifier, tr_loader, val_loader, epochs=10, max_patience=5, save_path="models/first_#104.pt"):
    best_loss = math.inf
    model.to(device)
    opt = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    loss = 0
    hist = {'tr': {'loss': [], 'top1': [], 'top3': []}, 'val': {'loss': [], 'top1': [], 'top3': []}}
    patience = 0
    for epoch in range(epochs):
        progbar = tqdm(tr_loader, total=len(tr_loader))
        progbar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        top1_acc, top3_acc = 0, 0
        loss_tot = 0
        model.train()
        for obj_mb, office_mb, label_mb, _ in tr_loader:
            opt.zero_grad()
            obj_mb = obj_mb.to(device)
            office_mb = office_mb.float().to(device)
            label_mb = label_mb.to(device)
            output = model(obj_mb, office_mb)
            top1_acc += sum(output.cpu().argmax(dim=1) == label_mb.cpu()) / len(label_mb) # WHY .cpu() ??
            top3_acc += sum([1 if label_mb[i] in torch.topk(output[i],3).indices else 0 for i in range(len(label_mb))]) / len(label_mb)
            loss = criterion(output, label_mb)
            loss.backward()
            opt.step()
            loss_tot += loss.item()
            # update progress bar
            progbar.update()
            progbar.set_postfix(tr_loss=f"{loss.item():.4f}")
        top1_acc /= len(tr_loader)
        top3_acc /= len(tr_loader)
        loss_tot /= len(tr_loader)
        last_tr_loss, last_tr_top1_acc, last_tr_top3_acc = str(f"{loss_tot:.4f}"), str(f"{top1_acc:.4f}"), str(f"{top3_acc:.4f}")   # used only for the progbar
        progbar.set_postfix(tr_loss=last_tr_loss, tr_top1_acc=last_tr_top1_acc, tr_top3_acc=last_tr_top3_acc)
        hist['tr']['top1'].append(top1_acc)
        hist['tr']['top3'].append(top3_acc)
        hist['tr']['loss'].append(loss_tot)
        torch.cuda.empty_cache()

        # validation
        val_loss, val_top1_acc, val_top3_acc = evaluate(model, val_loader, criterion)

        # progbar.update()
        progbar.set_postfix(tr_loss=last_tr_loss, tr_top1_acc=last_tr_top1_acc, tr_top3_acc=last_tr_top3_acc, val_loss=f"{val_loss:.4f}", val_top1_acc=f"{val_top1_acc:.4f}", val_top3_acc=f"{val_top3_acc:.4f}")
        progbar.close()

        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Model saved at {save_path} - epoch {epoch+1}/{epochs}")
            torch.save(model.cpu(), save_path)
            model.to(device)

        # patience
        hist['val']['top1'].append(val_top1_acc)
        hist['val']['top3'].append(val_top3_acc)
        hist['val']['loss'].append(val_loss)
        if len(hist['val']['loss']) > 1 and val_loss >= hist['val']['loss'][-2-patience]:
            patience += 1
            if patience >= max_patience:
                progbar.close()
                break
        else:
            patience = 0

        torch.cuda.empty_cache()

    return hist


def evaluate(model: CTTClassifier, loader: DataLoader, criterion):
    model.eval()
    top1_acc, top3_acc = 0, 0
    loss = 0
    for obj_mb, office_mb, label_mb, _ in loader:
        obj_mb = obj_mb.to(device)
        office_mb = office_mb.float().to(device)
        label_mb = label_mb.to(device)
        output = model(obj_mb, office_mb)
        top1_acc += sum(output.cpu().argmax(dim=1) == label_mb.cpu()) / len(label_mb) # WHY .cpu() ??
        top3_acc += sum([1 if label_mb[i] in torch.topk(output[i],3).indices else 0 for i in range(len(label_mb))]) / len(label_mb)
        loss += criterion(output, label_mb).item()
    top1_acc /= len(loader)
    top3_acc /= len(loader)
    loss /= len(loader)
    return loss, top1_acc, top3_acc

if __name__ == '__main__':
    seq_len = 30

    # read dataset csv and split it
    dataset = pd.read_csv('dataset/rebalanced_dataset.csv', sep='\t', index_col=0)
    train_set, val_set = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=1)
    train_set = BalancedCTTDataset(dataset=train_set, offices_path='dataset/offices_names.csv', seq_len=seq_len, task='FirstLevel')
    training_parser = train_set.parser
    valid_set = BalancedCTTDataset(dataset=val_set, offices_path='dataset/offices_names.csv', parser=training_parser, seq_len=seq_len, task='FirstLevel')
    with open("models/first_#104_parser", "wb") as f:
        pickle.dump({"parser": training_parser}, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_loader = DataLoader(train_set, batch_size=512, num_workers=2)
    val_loader = DataLoader(valid_set, batch_size=128, num_workers=2)
    model = CTTClassifier(left_dims=(len(train_set.parser.alphabet) + 2, 256), left_inp_dim=256, seq_len=seq_len)  # +1 because of token "<UNK>"
    hist = train(model, train_loader, val_loader, epochs=30)

    plt.subplot(1, 3, 1)
    plt.plot(hist['tr']['loss'], label='training loss')
    plt.plot(hist['val']['loss'], label='validation loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(hist['tr']['top1'], label='training top1 accuracy')
    plt.plot(hist['val']['top1'], label='validation top1 accuracy')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(hist['tr']['top3'], label='training top3 accuracy')
    plt.plot(hist['val']['top3'], label='validation top3 accuracy')
    plt.legend()
    plt.show()
