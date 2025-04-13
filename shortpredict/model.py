# 纯stid
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import os

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

class Ding(nn.Module):
    def __init__(self, 
                device,
                input_dim=3,
                channels=32,
                num_nodes=576,
                input_len=12,
                output_len=12,
                dropout=0.1, 
                ):
        super(Ding, self).__init__()
        input_dim = 2
        self.stid = STID(num_nodes=num_nodes,node_dim=channels,input_len=input_len,input_dim=input_dim,
                         embed_dim=channels,output_len=output_len,num_layer=3,temp_dim_diy=channels,day_in_year_size=366)
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    def forward(self, x): # B T N 3
        pred = self.stid(x)
        return pred
