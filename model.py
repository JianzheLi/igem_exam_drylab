import torch
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims: list = [2048, 512, 64], dropout = 0.1):
       
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = h_dim
        
        layers.append(nn.Linear(in_features, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        #  (batch_size, seq_len, d_model)
        batch_size = x.size(0)
        
        x = x.reshape(batch_size, -1)  
        
        return self.net(x)

class DNA_Transformer(nn.Module):
    def __init__(self, 
                 seq_length=70, 
                 n_features=4,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()
        #self.attention_weights = None
        self.embedding = nn.Sequential(
            nn.Linear(n_features, d_model),  #embedding
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        #position
        self.position_embed = nn.Parameter(
            torch.randn(1, seq_length, d_model)
        )
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
       
        self.classifier = Classifier(
            input_dim=seq_length * d_model,
            hidden_dims=[2048, 512, 64],
            dropout=dropout
        )

    def _record_attention(self, attn_weights):
       
        self.attention_weights = [aw.detach().cpu() for aw in attn_weights]
    
    def get_encode(self,x):
        x = self.embedding(x)  # (batch_size, len, d_model)
        x = x + self.position_embed 
        x = self.transformer(x)
        return x

    def forward(self, x):
        
        x = self.embedding(x)  # (batch_size, len, d_model)
        x = x + self.position_embed 

        """handles = []
        def hook(module, args, output):
            # output: (output, attention_weights)
            if len(output) >= 2 and output[1] is not None:
                self._record_attention(output[1])
        
        
        for layer in self.transformer.layers:
            handle = layer.self_attn.register_forward_hook(hook)
            handles.append(handle)"""

        #x = x.permute(1, 0, 2)  # (len, batch_size, d_model)
        
        x = self.transformer(x)  # (batch_size,len,d_model)
        #x = x.permute(1, 0, 2)  #  (batch_size, len, d_model)

        
        #print(x.shape)
        
        return self.classifier(x)
    

class DNA_LSTM(nn.Module):
    def __init__(self, 
                 seq_length=70,
                 n_features=4,
                 d_model=128,
                 num_layers=4,
                 dropout=0.1,
                 bidirectional=True):
        super().__init__()
        
      
        self.embedding = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
     
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model//2 if bidirectional else d_model,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        
        self.classifier = Classifier(
            input_dim=seq_length * d_model,
            hidden_dims=[2048, 512, 64],
            dropout=dropout
        )
        
    
        self._init_weights()
        
    def _init_weights(self):
        """ Xavier初始化 """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

    def get_encode(self, x):
       
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs  # [batch, seq_len, d_model*num_directions]

    def forward(self, x):
      
        x = self.embedding(x)  # [batch, seq_len, d_model]
        
    
        x, (h_n, c_n) = self.lstm(x)  # x: [batch, seq_len, d_model*num_directions]
        
        return self.classifier(x)
