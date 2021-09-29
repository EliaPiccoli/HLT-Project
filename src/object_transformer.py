import torch
from torch import nn
from torch._C import dtype
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from data_parser import Parser 
from dataset import BalancedCTTDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, dims=(30, 512), nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(dims[0], dims[1])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dims[1], nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        tensor_shape = x.shape
        mask = (x == 1).reshape(x.shape).to(device)
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x


def train(model: Transformer, loader: DataLoader, data: torch.Tensor, epochs=10):
    model.to(device)
    model.train()
    data = data.to(device)
    for epoch in range(epochs):
        for obj_mb, office_mb, label_mb in loader:
            obj_mb = obj_mb.to(device)
            outputs = model(obj_mb)
            print(outputs.shape)
            exit()


def get_data() -> torch.Tensor:
    data = pd.read_csv('../dataset/balanced_dataset.csv', sep='\t', index_col=0)
    data_obj = data['Oggetto'].to_numpy()
    parser = Parser()
    parser.create_vocab(data_obj)
    encoded_data_obj = parser.encode_data(data_obj)
    return torch.autograd.Variable(torch.Tensor(encoded_data_obj).long())


if __name__ == '__main__':
    data = get_data()
    # print(data.shape, data[0], torch.max(data).item())
    training_set = BalancedCTTDataset(dataset='dataset/balanced_dataset.csv', offices_path='../dataset/offices_names.csv')
    training_generator = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
    model = Transformer(dims=(torch.max(data).item(),512, 15))
    train(model, training_generator, data[0], epochs=1)