#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入模型
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from shortpredict.long.model_agcrn import AGCRN

def load_data(data_path, scaler_path, index_path):
    """
    加载测试数据和归一化参数
    """
    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    processed_data = data['processed_data']
    
    # 加载索引
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    test_index = index['test']
    
    # 加载归一化参数
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return processed_data, test_index, scaler

def prepare_test_data(processed_data, test_index, input_len, output_len):
    """
    准备测试数据
    """
    x_test, y_test = [], []
    for idx in test_index:
        history_start, history_end, future_end = idx
        x = processed_data[history_start:history_end]
        y = processed_data[history_end:future_end, :, :2]  # 只取前两个特征（海温、海盐）
        x_test.append(x)
        y_test.append(y)
    
    x_test = torch.FloatTensor(np.array(x_test))  # [B, T, N, C]
    y_test = torch.FloatTensor(np.array(y_test))  # [B, T', N, 2]
    
    return x_test, y_test

def load_model(model_path, params):
    """
    加载训练好的模型
    """
    device = torch.device(params['device'] if torch.cuda.is_available() else "cpu")
    
    model = AGCRN(
        device=device,
        input_dim=params['input_dim'],
        channels=params['channels'],
        num_nodes=params['num_nodes'],
        input_len=params['input_len'],
        output_len=params['output_len'],
        dropout=params['dropout'],
        cheb_k=params['cheb_k'],
        embed_dim=params['embed_dim'],
        num_layers=params['num_layers']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def inverse_transform(data, scaler):
    """
    反归一化数据
    """
    mean, std = scaler['args']['mean'], scaler['args']['std']
    
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).type_as(data).to(data.device).view(1, 1, -1)
        std = torch.from_numpy(std).type_as(data).to(data.device).view(1, 1, -1)
    
    data = data * std
    data = data + mean
    
    return data

def reshape_to_grid(data, grid_size=24):
    """
    将一维节点列表重塑为二维网格形状，用于热力图可视化
    """
    # data shape: [N, C] -> [grid_size, grid_size, C]
    return data.reshape(grid_size, grid_size, -1)

def visualize_heatmaps(y_true, y_pred, sample_idx, time_idx, feature_idx, feature_name, grid_size=24, save_path=None):
    """
    可视化真实值和预测值的热力图
    
    参数:
        y_true: 真实值 [B, T, N, C]
        y_pred: 预测值 [B, T, N, C]
        sample_idx: 要可视化的样本索引
        time_idx: 要可视化的时间步索引
        feature_idx: 要可视化的特征索引（0:海温, 1:海盐）
        feature_name: 特征名称（用于标题）
        grid_size: 网格尺寸（默认为24x24）
        save_path: 保存路径，若不为None则保存图片
    """
    # 获取指定样本、时间步和特征的数据
    true_data = y_true[sample_idx, time_idx, :, feature_idx].cpu().numpy()
    pred_data = y_pred[sample_idx, time_idx, :, feature_idx].cpu().numpy()
    
    # 计算误差
    error_data = np.abs(true_data - pred_data)
    
    # 重塑为网格形状
    true_grid = reshape_to_grid(true_data, grid_size)
    pred_grid = reshape_to_grid(pred_data, grid_size)
    error_grid = reshape_to_grid(error_data, grid_size)
    
    # 创建自定义的颜色映射和标准化范围
    vmin = min(true_data.min(), pred_data.min())
    vmax = max(true_data.max(), pred_data.max())
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制真实值热力图
    im1 = axes[0].imshow(true_grid[:, :, 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'真实{feature_name}')
    fig.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 绘制预测值热力图
    im2 = axes[1].imshow(pred_grid[:, :, 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'预测{feature_name}')
    fig.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 绘制误差热力图
    im3 = axes[2].imshow(error_grid[:, :, 0], cmap='Reds')
    axes[2].set_title(f'绝对误差')
    fig.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # 设置整体标题
    plt.suptitle(f'样本 {sample_idx+1}, 预测第 {time_idx+1} 个月的{feature_name}对比', fontsize=16)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存图片至: {save_path}")
    
    plt.show()

def create_animation_frames(y_true, y_pred, sample_idx, feature_idx, feature_name, grid_size=24, output_dir=None):
    """
    为创建动画准备一系列热力图帧
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    n_timesteps = y_true.shape[1]
    
    for t in range(n_timesteps):
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"{feature_name}_sample{sample_idx}_time{t+1}.png")
        
        visualize_heatmaps(
            y_true, y_pred, sample_idx, t, feature_idx, 
            f"{feature_name} (月份 {t+1})", grid_size, save_path
        )

def main():
    parser = argparse.ArgumentParser(description='AGCRN模型测试可视化')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录，包含参数和模型文件')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--sample_idx', type=int, default=0, help='要可视化的样本索引')
    parser.add_argument('--feature', type=str, choices=['temperature', 'salinity', 'both'], default='both', help='要可视化的特征')
    parser.add_argument('--output_dir', type=str, default='/Users/huiyangzheng/Desktop/Project/统计建模大赛/statistical-modeling/analysis_plots/test_predict_compare', help='输出目录')
    parser.add_argument('--animation', action='store_true', help='是否创建动画帧')
    parser.add_argument('--grid_size', type=int, default=24, help='网格尺寸')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载模型参数
    param_files = [f for f in os.listdir(args.model_dir) if f.startswith('params') and f.endswith('.json')]
    if not param_files:
        raise FileNotFoundError("在指定目录中未找到参数文件")
    
    with open(os.path.join(args.model_dir, param_files[0]), 'r') as f:
        params = json.load(f)
    
    # 构建数据文件路径
    input_len = params['input_len']
    output_len = params['output_len']
    
    prefix = "monthly_" if params.get('use_monthly', False) else ""
    data_path = os.path.join(args.data_dir, f"{prefix}data_in_{input_len}_out_{output_len}.pkl")
    index_path = os.path.join(args.data_dir, f"{prefix}index_in_{input_len}_out_{output_len}.pkl")
    scaler_path = os.path.join(args.data_dir, f"scaler_in_{input_len}_out_{output_len}.pkl")
    
    # 找到最佳模型文件
    model_file = os.path.join(args.model_dir, "best_model.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError("在指定目录中未找到best_model.pth")
    
    # 加载数据
    processed_data, test_index, scaler = load_data(data_path, scaler_path, index_path)
    
    # 准备测试数据
    x_test, y_test = prepare_test_data(processed_data, test_index, input_len, output_len)
    
    # 加载模型
    model, device = load_model(model_file, params)
    
    # 进行预测
    with torch.no_grad():
        x_test = x_test.to(device)
        y_pred = model(x_test)
    
    # 反归一化数据
    y_test = inverse_transform(y_test, scaler)
    y_pred = inverse_transform(y_pred, scaler)
    
    # 将数据移回CPU
    y_test = y_test.cpu()
    y_pred = y_pred.cpu()
    
    # 开始可视化
    sample_idx = min(args.sample_idx, len(test_index) - 1)
    feature_map = {'temperature': 0, 'salinity': 1}
    
    if args.animation:
        # 创建动画帧
        if args.feature == 'both' or args.feature == 'temperature':
            create_animation_frames(
                y_test, y_pred, sample_idx, 0, '海温', 
                args.grid_size, os.path.join(args.output_dir, 'temperature_frames')
            )
        
        if args.feature == 'both' or args.feature == 'salinity':
            create_animation_frames(
                y_test, y_pred, sample_idx, 1, '海盐', 
                args.grid_size, os.path.join(args.output_dir, 'salinity_frames')
            )
    else:
        # 可视化特定时间步
        mid_timestep = output_len // 2
        
        if args.feature == 'both' or args.feature == 'temperature':
            visualize_heatmaps(
                y_test, y_pred, sample_idx, mid_timestep, 0, '海温', 
                args.grid_size, os.path.join(args.output_dir, f'temperature_month{mid_timestep+1}_sample{sample_idx}.png')
            )
        
        if args.feature == 'both' or args.feature == 'salinity':
            visualize_heatmaps(
                y_test, y_pred, sample_idx, mid_timestep, 1, '海盐', 
                args.grid_size, os.path.join(args.output_dir, f'salinity_month{mid_timestep+1}_sample{sample_idx}.png')
            )

if __name__ == "__main__":
    main() 