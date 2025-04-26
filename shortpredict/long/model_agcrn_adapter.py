import torch
import torch.nn as nn
from .model_agcrn import AGCRN

class AGCRNAdapter(nn.Module):
    """
    适配器类，使AGCRN模型能够与原有的训练框架兼容
    """
    def __init__(self, 
                device,
                input_dim=3,  # 海温，海盐，日期
                channels=64,  
                num_nodes=576, 
                input_len=12,
                output_len=12,
                dropout=0.1,
                cheb_k=3,     # Chebyshev多项式阶数
                embed_dim=10,  # 节点嵌入维度
                num_layers=2   # DCRNN层数
                ):
        super(AGCRNAdapter, self).__init__()
        
        # 初始化AGCRN模型
        self.agcrn = AGCRN(
            device=device,
            input_dim=input_dim,
            channels=channels,
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            dropout=dropout,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            num_layers=num_layers
        )
        
    def param_num(self):
        """
        返回模型参数数量，与原训练框架兼容
        """
        return self.agcrn.param_num()
        
    def forward(self, x):
        """
        模型前向传播
        Args:
            x: 输入张量 [B, T, N, C] - batch, time_steps, nodes, features
        Returns:
            output: 预测结果 [B, T, N, C_out]
        """
        return self.agcrn(x) 