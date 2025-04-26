# 纯stid
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import os
import psutil
def get_process_memory():
    """获取当前进程内存使用(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

class BaseModel(nn.Module):
    """基础模型类，定义通用接口"""
    def __init__(self, num_nodes, input_dim, output_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        """前向传播
        Args:
            x: 输入张量 [B, T, N, C]
        Returns:
            预测结果 [B, T, N, C]
        """
        raise NotImplementedError
        
    def reshape_for_ml(self, x):
        """将时空数据重塑为机器学习模型可用的格式
        Args:
            x: 输入张量 [B, T, N, C]
        Returns:
            重塑后的数据 [B*N, T*C]
        """
        batch_size, seq_len, num_nodes, channels = x.shape
        x = x.permute(0, 2, 1, 3)  # [B, N, T, C]
        x = x.reshape(batch_size * num_nodes, -1)  # [B*N, T*C]
        return x

class MLPModel(BaseModel):
    """多层感知机模型实现"""
    def __init__(self, num_nodes, seq_len, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__(num_nodes, input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 构建MLP层
        layers = []
        # 输入层
        layers.append(nn.Linear(seq_len * input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, seq_len * output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size, seq_len, num_nodes, channels = x.shape
        
        # 重塑输入以适应MLP
        x = x.permute(0, 2, 1, 3)  # [B, N, T, C]
        x = x.reshape(batch_size, num_nodes, -1)  # [B, N, T*C]
        
        # MLP处理
        out = self.mlp(x)  # [B, N, seq_len*output_dim]
        
        # 重塑输出
        out = out.reshape(batch_size, num_nodes, seq_len, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # [B, T, N, output_dim]
        
        return out

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

class STID(BaseModel):
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
        super().__init__(num_nodes, input_dim, input_dim)
        # attributes
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = input_dim - 1 # 输入维度减去日期
        self.out_dim = input_dim - 1
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
    
class LSTMModel(BaseModel):
    """LSTM模型实现"""
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__(num_nodes, input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size, seq_len, num_nodes, channels = x.shape
        
        # 重塑输入以适应LSTM
        x = x.permute(0, 2, 1, 3)  # [B, N, T, C]
        x = x.reshape(batch_size * num_nodes, seq_len, channels)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 预测
        out = self.fc(lstm_out)  # [B*N, T, output_dim]
        
        # 重塑输出
        out = out.reshape(batch_size, num_nodes, seq_len, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # [B, T, N, C]
        
        return out

class GRUModel(BaseModel):
    """GRU模型实现"""
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__(num_nodes, input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size=input_dim,
                         hidden_size=hidden_dim,
                         num_layers=num_layers,
                         batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size, seq_len, num_nodes, channels = x.shape
        
        # 重塑输入以适应GRU
        x = x.permute(0, 2, 1, 3)  # [B, N, T, C]
        x = x.reshape(batch_size * num_nodes, seq_len, channels)
        
        # GRU处理
        gru_out, _ = self.gru(x)
        
        # 预测
        out = self.fc(gru_out)  # [B*N, T, output_dim]
        
        # 重塑输出
        out = out.reshape(batch_size, num_nodes, seq_len, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # [B, T, N, C]
        
        return out

class Ding(nn.Module):
    """模型包装器，可以选择不同的模型类型"""
    def __init__(self, 
                device,
                model_type='stid',  # 可选: 'stid', 'lstm', 'gru', 'mlp'
                input_dim=3,
                channels=32,
                num_nodes=576,
                input_len=12,
                output_len=12,
                dropout=0.1,
                **kwargs):
        super(Ding, self).__init__()
        self.device = device
        self.model_type = model_type
        
        if model_type == 'stid':
            self.model = STID(
                num_nodes=num_nodes,
                node_dim=channels,
                input_len=input_len,
                input_dim=input_dim,
                embed_dim=channels,
                output_len=output_len,
                num_layer=3,
                temp_dim_diy=channels,
                day_in_year_size=366
            )
        elif model_type == 'lstm':
            self.model = LSTMModel(
                num_nodes=num_nodes,
                input_dim=input_dim,
                hidden_dim=channels,
                num_layers=3,
                output_dim=input_dim - 1
            )
        elif model_type == 'gru':
            self.model = GRUModel(
                num_nodes=num_nodes,
                input_dim=input_dim,
                hidden_dim=channels,
                num_layers=3,
                output_dim=input_dim - 1
            )
        elif model_type == 'mlp':
            self.model = MLPModel(
                num_nodes=num_nodes,
                seq_len=input_len,
                input_dim=input_dim,
                hidden_dim=channels,
                num_layers=3,
                output_dim=input_dim - 1
            )
            
        self.model.to(device)
        
    def forward(self, x):
        # 在每个epoch开始时记录内存
        epoch_start_mem = get_process_memory()
        y = self.model(x)
        # 在每个epoch结束时记录内存
        epoch_end_mem = get_process_memory()
        train_memory = epoch_end_mem - epoch_start_mem
        print(f"Training Memory: {train_memory:.2f} MB")
        return y
        
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
