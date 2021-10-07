import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, dims=(30, 512), nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(dims[0], dims[1])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dims[1], nhead=nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        mask = (x == 1).reshape(x.shape).to(device)
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x