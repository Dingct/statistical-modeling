from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import os

class SpaceEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(SpaceEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        
        self.SpaceConv = nn.Conv2d(in_channels=c_in, out_channels=c_in, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        B, C, _= x.shape
        x=x.reshape(B,C,-1,360)
        #x=x.reshape(B,C,-1,1440)
        x = self.SpaceConv(x)
        x=x.reshape(B,C,-1)
        return x

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, 
                 num_nodes,
                 node_dim,
                 input_len,
                 input_dim,
                 embed_dim,
                 output_len,
                 num_layer,
                 temp_dim_diy,
                 day_in_year_size):
        super().__init__()
        # attributes
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = input_dim
        self.out_dim = input_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.num_layer = num_layer
        self.temp_dim_diy = temp_dim_diy
        self.day_in_year_size = day_in_year_size

        # spatial embeddings
        self.node_emb = nn.Parameter(
            torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # temporal embeddings
        self.day_in_year_emb = nn.Parameter(
            torch.empty(self.day_in_year_size, self.temp_dim_diy))
        nn.init.xavier_uniform_(self.day_in_year_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim + self.temp_dim_diy
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len*self.out_dim, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        d_i_y_data = history_data[..., self.input_dim]
        day_in_year_emb = self.day_in_year_emb[(d_i_y_data[:, -1, :] * self.day_in_year_size).type(torch.LongTensor)]

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        # expand node embeddings
        node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        
        # temporal embeddings
        tem_emb = []
        tem_emb.append(day_in_year_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden).transpose(1, 2).contiguous().view(batch_size, num_nodes, self.output_len, self.out_dim).transpose(1, 2)
        return prediction

# Multifactor Correlation Learning + Factor-Specific Prediction Towers
class Mix(nn.Module):
    def __init__(self, d_model, conv_dff, out_len, c_out, dropout=0.1):
        super(Mix, self).__init__()
        # 来自model_Informer.py
        self.pred_len = out_len
        self.ffn2pw1 = nn.Conv1d(in_channels=2 * d_model, out_channels=d_model * conv_dff, kernel_size=1, stride=1,padding=0, dilation=1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=d_model * conv_dff, out_channels=2 * d_model , kernel_size=1, stride=1,padding=0, dilation=1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2drop2 = nn.Dropout(dropout)
        self.mlp_one = nn.Linear(d_model*2, c_out, bias=True)
        self.mlp_two = nn.Linear(d_model*2, c_out, bias=True)
    def forward(self, dec_one,dec_two): # N个B,L,D 例如：2个batch_size,seq_len,d_model
        all=torch.stack((dec_one,dec_two),axis=3)
        B, L, D, N = all.shape
        all = all.permute(0, 2, 3, 1) # B,D,N,L
        all = all.reshape(B, D * N, L)
        all = self.ffn2drop1(self.ffn2pw1(all)) # B,D*d,L
        all = self.ffn2act(all)
        all = self.ffn2drop2(self.ffn2pw2(all)) # B,D*N,L
        all = all.permute(0, 2, 1) # B,L,D*N
        dec_out_one=self.mlp_one(all) # B,L,C
        dec_out_two=self.mlp_two(all)
        dec_out_coarse=torch.stack((dec_out_one,dec_out_two),axis=3) 
        return dec_out_coarse[:, -self.pred_len:, :,:] # B,L,C,N 例如：batch_size,seq_len,c_out,2

class Ding(nn.Module):
    def __init__(self, 
                device,
                input_dim=3,
                channels=64,
                num_nodes=576,
                input_len=12,
                output_len=12,
                dropout=0.1, 
                ):
        super(Ding, self).__init__()
        input_dim = 2
        self.stid = STID(num_nodes=num_nodes,node_dim=channels,input_len=input_len,input_dim=input_dim,
                         embed_dim=channels,output_len=output_len,num_layer=2,temp_dim_diy=channels,day_in_year_size=366)
        self.mix = Mix(d_model=channels, conv_dff=2, out_len=output_len, c_out=input_dim,dropout=dropout)
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    def forward(self, x): # B T N 3
        # t = x[...,2:]
        # x1 = self.stid(torch.cat([x[...,0:1]]+[t],dim=-1))[...,0]
        # x2 = self.stid(torch.cat([x[...,1:2]]+[t],dim=-1))[...,0] # B T N
        # pred = self.mix(x1,x2)
        pred = self.stid(x)
        return pred
