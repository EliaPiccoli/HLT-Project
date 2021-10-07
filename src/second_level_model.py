import torch
import math
import pickle
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from object_transformer import Transformer
from first_level_model import FirstLevelClassifier
from dataset import BalancedCTTDataset
from sklearn.model_selection import train_test_split

# set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SecondLevelClassifier(nn.Module):
    def __init__(self, obj_transformer_size=(20000, 128), output_dim=118, object_final_dim=128, seq_len=30, nhead=8, dim_feedforward=2048,
                 transformer_dropout=0.7, decoder_dropout=0.2, activation='relu', num_layers=2, index_level_map=None):
        super().__init__()
        # Office module
        office_onehot = 20
        
        # First Level module
        self.first_level_module = torch.load('models/first_####.pt', map_location=torch.device(device))
        for param in self.first_level_module.parameters():
            param.requires_grad = False
        
        # Second Level module
        self.output_dim = output_dim # number of classes
        self.object_module = Transformer(dims=obj_transformer_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=transformer_dropout, activation=activation, num_layers=num_layers)
        self.final_object = nn.Linear(seq_len*obj_transformer_size[1], object_final_dim)
        # Decoder with 1 single layer
        self.decoder = nn.Sequential(nn.Linear(object_final_dim + office_onehot, output_dim))
        self.dropout = nn.Dropout(decoder_dropout)

        if index_level_map is None:
            print("No class-index dict found..")
            exit()
        else:
            self.idx2lvl = index_level_map

    def forward(self, obj, off):
        first_level_out = self.first_level_module(obj, off)
        enc_obj = self.object_module(obj)
        enc_obj = torch.flatten(enc_obj, start_dim=1)
        final_obj = F.relu(self.final_object(enc_obj))
        final_enc = torch.cat((final_obj, off), dim=1)
        x = self.dropout(final_enc)
        x = self.decoder(x)
        tensor_shape = x.shape

        # replicate first level predictions
        z = torch.zeros(tensor_shape)
        if x.is_cuda:
            z = z.to(device)
        for i in range(first_level_out.shape[1]): 
            indexes = torch.LongTensor(self.idx2lvl[i])
            if x.is_cuda:
                indexes = indexes.to(device)
            slice = first_level_out[:,i].reshape(-1,1).repeat(1,len(indexes))
            z.index_copy_(1,indexes,slice)
        # apply first level bias
        y = x * z

        assert(y.shape == x.shape)
        return y

class SecondLevelNoBiasClassifier(nn.Module):
    def __init__(self, obj_transformer_size=(20000, 128), output_dim=118, object_final_dim=128, seq_len=30, nhead=8, dim_feedforward=2048,
                 transformer_dropout=0.7, decoder_dropout=0.2, activation='relu', num_layers=2):
        super().__init__()
        # Office module
        office_onehot = 20
        
        # Second Level module
        self.output_dim = output_dim # number of classes
        self.object_module = Transformer(dims=obj_transformer_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=transformer_dropout, activation=activation, num_layers=num_layers)
        self.final_object = nn.Linear(seq_len*obj_transformer_size[1], object_final_dim)

        # Decoder with 1 single layer
        self.decoder = nn.Sequential(nn.Linear(object_final_dim + office_onehot, output_dim))
        self.dropout = nn.Dropout(decoder_dropout)

    def forward(self, obj, off):
        enc_obj = self.object_module(obj)
        enc_obj = torch.flatten(enc_obj, start_dim=1)
        final_obj = F.relu(self.final_object(enc_obj))
        final_enc = torch.cat((final_obj, off), dim=1)
        x = self.dropout(final_enc)
        x = self.decoder(x)
        return x

def train(model, tr_loader, val_loader, epochs=10, max_patience=5, save_path="models/second_####.pt"):
    best_loss = math.inf
    model.to(device)
    opt = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    loss = 0
    hist = {'tr': {'loss': [], 'top1': [], 'top3': [], 'top5': []}, 'val': {'loss': [], 'top1': [], 'top3': [], 'top5': []}}
    patience = 0
    # training cycle
    for epoch in range(epochs):
        progbar = tqdm(tr_loader, total=len(tr_loader))
        progbar.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        top1_acc, top3_acc, top5_acc = 0, 0, 0
        loss_tot = 0
        model.train()
        for obj_mb, office_mb, label_mb, _ in tr_loader:
            opt.zero_grad()
            obj_mb = obj_mb.to(device)
            office_mb = office_mb.float().to(device)
            label_mb = label_mb.to(device)
            output = model(obj_mb, office_mb)
            top1_acc += sum(output.cpu().argmax(dim=1) == label_mb.cpu()) / len(label_mb) 
            top3_acc += sum([1 if label_mb[i] in torch.topk(output[i],3).indices else 0 for i in range(len(label_mb))]) / len(label_mb)
            top5_acc += sum([1 if label_mb[i] in torch.topk(output[i],5).indices else 0 for i in range(len(label_mb))]) / len(label_mb)
            loss = criterion(output, label_mb)
            loss.backward()
            opt.step()
            loss_tot += loss.item()
            # update progress bar
            progbar.update()
            progbar.set_postfix(tr_loss=f"{loss.item():.4f}")
        top1_acc /= len(tr_loader)
        top3_acc /= len(tr_loader)
        top5_acc /= len(tr_loader)
        loss_tot /= len(tr_loader)
        last_tr_loss, last_tr_top1_acc, last_tr_top3_acc, last_tr_top5_acc = str(f"{loss_tot:.4f}"), str(f"{top1_acc:.4f}"), str(f"{top3_acc:.4f}"), str(f"{top5_acc:.4f}")
        progbar.set_postfix(tr_loss=last_tr_loss, tr_top1_acc=last_tr_top1_acc, tr_top3_acc=last_tr_top3_acc, tr_top5_acc=last_tr_top5_acc) 
        hist['tr']['top1'].append(top1_acc)
        hist['tr']['top3'].append(top3_acc)
        hist['tr']['top5'].append(top5_acc)
        hist['tr']['loss'].append(loss_tot)
        torch.cuda.empty_cache()

        # validation
        val_loss, val_top1_acc, val_top3_acc, val_top5_acc = evaluate(model, val_loader, criterion)

        # progbar.update()
        progbar.set_postfix(tr_loss=last_tr_loss, tr_top1_acc=last_tr_top1_acc, tr_top3_acc=last_tr_top3_acc, tr_top5_acc=last_tr_top5_acc, 
                    val_loss=f"{val_loss:.4f}", val_top1_acc=f"{val_top1_acc:.4f}", val_top3_acc=f"{val_top3_acc:.4f}", val_top5_acc=f"{val_top5_acc:.4f}")
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
        hist['val']['top5'].append(val_top5_acc)
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

# validation
def evaluate(model: SecondLevelClassifier, loader: DataLoader, criterion):
    model.eval()
    top1_acc, top3_acc, top5_acc = 0, 0, 0
    loss = 0
    for obj_mb, office_mb, label_mb, _ in loader:
        obj_mb = obj_mb.to(device)
        office_mb = office_mb.float().to(device)
        label_mb = label_mb.to(device)
        output = model(obj_mb, office_mb)
        top1_acc += sum(output.cpu().argmax(dim=1) == label_mb.cpu()) / len(label_mb) 
        top3_acc += sum([1 if label_mb[i] in torch.topk(output[i],3).indices else 0 for i in range(len(label_mb))]) / len(label_mb)
        top5_acc += sum([1 if label_mb[i] in torch.topk(output[i],5).indices else 0 for i in range(len(label_mb))]) / len(label_mb)
        loss += criterion(output, label_mb).item()
    top1_acc /= len(loader)
    top3_acc /= len(loader)
    top5_acc /= len(loader)
    loss /= len(loader)
    return loss, top1_acc, top3_acc, top5_acc

if __name__ == '__main__':
    seq_len = 95

    # read dataset
    dataset = pd.read_csv('dataset/rebalanced_dataset.csv', sep='\t', index_col=0)

    # Load model parser
    dc = {}
    with open("models/first_####_parser", "rb") as f:
        dc = pickle.load(f)
    first_lvl_model_parser = dc["parser"]
    # split data in training and validation set
    train_set, val_set = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=1)
    train_set = BalancedCTTDataset(dataset=train_set, offices_path='dataset/offices_names.csv', parser=first_lvl_model_parser, task="SecondLevel", seq_len=seq_len)
    valid_set = BalancedCTTDataset(dataset=val_set, offices_path='dataset/offices_names.csv', parser=first_lvl_model_parser, task="SecondLevel", seq_len=seq_len)

    # load first2second level map
    first_second_class_mapping_dict = {}
    with open("models/class_to_label", "rb") as f:
        first_second_class_mapping_dict = pickle.load(f)
    class_mapping_dict = first_second_class_mapping_dict['class_mapping']
    # create data loaders
    train_loader = DataLoader(train_set, batch_size=512, num_workers=2)
    val_loader = DataLoader(valid_set, batch_size=128, num_workers=2)
    # create and train the model
    model = SecondLevelClassifier(obj_transformer_size=(len(first_lvl_model_parser.alphabet) + 2, 256), object_final_dim=256, seq_len=seq_len, index_level_map=class_mapping_dict)  # +2 because of tokens <UNK> and <PAD>
    hist = train(model, train_loader, val_loader, epochs=20)
    
    # plot results
    plt.subplot(1, 4, 1)
    plt.plot(hist['tr']['loss'], label='training loss')
    plt.plot(hist['val']['loss'], label='validation loss')
    plt.legend()
    plt.subplot(1, 4, 2)
    plt.plot(hist['tr']['top1'], label='training top1 accuracy')
    plt.plot(hist['val']['top1'], label='validation top1 accuracy')
    plt.legend()
    plt.subplot(1, 4, 3)
    plt.plot(hist['tr']['top3'], label='training top3 accuracy')
    plt.plot(hist['val']['top3'], label='validation top3 accuracy')
    plt.legend()
    plt.subplot(1, 4, 4)
    plt.plot(hist['tr']['top5'], label='training top5 accuracy')
    plt.plot(hist['val']['top5'], label='validation top5 accuracy')
    plt.legend()
    plt.show()