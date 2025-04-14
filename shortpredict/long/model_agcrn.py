import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AVWGCN(nn.Module):
    """
    Adaptive Vertex-weighted Graph Convolutional Network
    """
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)
        
    def forward(self, x, node_embeddings):
        # x shaped [B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        # Adaptive adjacency matrix
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # Chebyshev polynomials approximation: T_0(L)=I, T_1(L)=L, T_k(L)=2LT_{k-1}(L)-T_{k-2}(L)
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # Adaptive weight matrix
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                      # N, dim_out
        # Graph convolution
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # B, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    """
    Adaptive Graph Convolutional Recurrent Network Cell
    """
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        # Gate mechanism with graph convolution
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        # Gated mechanism similar to GRU
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class AVWDCRNN(nn.Module):
    """
    Adaptive Vertex-weighted Diffusion Convolutional Recurrent Neural Network
    """
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        
        # 检查输入形状，确保N维是否需要转置
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 修改为更灵活的条件，可以处理不同形状的输入
        input_dim = x.shape[-1]  # 获取输入特征维度
        
        # 重塑输入为期望的形状 (B, T, N, D)
        if x.shape[2] != self.node_num:
            if x.shape[2] * x.shape[3] == self.node_num * self.input_dim:
                # 如果总元素数量匹配，重塑
                x = x.reshape(batch_size, seq_len, self.node_num, self.input_dim)
            else:
                # 如果输入维度不匹配，尝试不同的处理
                # 例如，可能需要变换输入顺序，或使用自适应层
                x = x.reshape(batch_size, seq_len, self.node_num, -1)
                if x.shape[3] != self.input_dim:
                    # 如果特征维度不匹配，使用线性投影
                    adapter = nn.Linear(x.shape[3], self.input_dim).to(x.device)
                    x = adapter(x)
        
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_len):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      # (num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    """
    Adaptive Graph Convolutional Recurrent Network for Spatiotemporal Prediction
    """
    def __init__(self, 
                device,
                input_dim=3,  # 海温，海盐，日期
                channels=64,  # equivalent to rnn_units
                num_nodes=576,  # 默认24*24
                input_len=12,
                output_len=12,
                dropout=0.1,
                cheb_k=3,  # Chebyshev多项式阶数
                embed_dim=10,  # 节点嵌入维度
                num_layers=2  # DCRNN层数
                ):
        super(AGCRN, self).__init__()
        self.output_dim = 2  # 输出海温和海盐
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        
        # 节点嵌入，学习节点之间的关系
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, embed_dim), requires_grad=True)
        
        # 编码器
        self.encoder = AVWDCRNN(num_nodes, input_dim, channels, 
                                cheb_k, embed_dim, num_layers)
        
        # 预测层 - 使用一个卷积层进行时间维度上的映射
        self.end_conv = nn.Conv2d(1, output_len * self.output_dim, 
                                 kernel_size=(1, channels), bias=True)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化参数
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
        
    def forward(self, x):
        # x shape: [B, T, N, C] - batch, time_steps, nodes, features
        
        # 获取数据的各维度大小
        batch_size = x.shape[0]
        
        # 确保输入维度正确
        # 如果需要，对输入进行重塑或特征选择
        if x.dim() == 4:
            # 如果输入是4维的 [B, T, N, C]
            if x.shape[-1] != self.input_dim and x.shape[-1] >= self.output_dim:
                # 如果输入特征数量不等于模型输入维度，但至少包含输出维度
                inputs = x[..., :self.input_dim] if x.shape[-1] >= self.input_dim else x
            else:
                inputs = x
        else:
            # 如果输入不是4维的，尝试重塑
            print(f"Warning: Input shape {x.shape} is not 4D, attempting to reshape.")
            if x.dim() == 3:  # [B, T, N*C]
                inputs = x.reshape(batch_size, -1, self.num_nodes, self.input_dim)
            else:
                # 其他情况，打印错误并返回空结果
                print(f"Error: Cannot process input shape {x.shape}")
                return torch.zeros(batch_size, self.output_len, self.num_nodes, self.output_dim).to(x.device)
        
        # 初始化隐藏状态
        init_state = self.encoder.init_hidden(batch_size).to(self.device)
        
        # 编码器前向传播
        output, _ = self.encoder(inputs, init_state, self.node_embeddings)
        
        # 取最后一个时间步的输出作为预测的起点
        output = output[:, -1:, :, :]  # [B, 1, N, hidden_dim]
        
        # 通过卷积层映射到预测时间步长度
        output = self.dropout(output)
        output = self.end_conv(output)  # [B, output_len*output_dim, N, 1]
        
        # 重塑为最终输出形状
        output = output.squeeze(-1).reshape(batch_size, self.output_len, self.num_nodes, self.output_dim)
        
        # 调整维度顺序 [B, T, N, C]
        return output

