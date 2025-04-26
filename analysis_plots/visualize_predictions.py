#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入模型
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from shortpredict.long.model_agcrn import AGCRN

def load_data(data_dir, input_len=12, output_len=6):
    """
    加载测试数据和归一化参数
    
    参数:
        data_dir (str): 数据目录
        input_len (int): 输入序列长度，默认为12月
        output_len (int): 输出序列长度，默认为6月
    
    返回:
        tuple: (processed_data, test_index, scaler)
    """
    # 构建文件路径
    data_path = os.path.join(data_dir, f"monthly_data_in_{input_len}_out_{output_len}.pkl")
    index_path = os.path.join(data_dir, f"monthly_index_in_{input_len}_out_{output_len}.pkl")
    scaler_path = os.path.join(data_dir, f"scaler_in_{input_len}_out_{output_len}.pkl")
    
    # 验证文件存在
    for file_path in [data_path, index_path, scaler_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
    
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
    
    print(f"已加载数据: 形状={processed_data.shape}, 测试样本数={len(test_index)}")
    
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
    
    print(f"测试数据形状: x={x_test.shape}, y={y_test.shape}")
    
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
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device)
    
    # 处理可能的前缀差异
    if all(k.startswith('agcrn.') for k in state_dict.keys()):
        # 移除 'agcrn.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('agcrn.', '')] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"已加载模型: {model_path}")
    
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
    
    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Increased figsize slightly

    # Define coordinate ranges (adjust if needed, maybe get from data source later)
    lon_min, lon_max = 122.0, 128.0
    lat_min, lat_max = 26.0, 32.0
    # extent format: [left, right, bottom, top]
    # imshow plots origin (0,0) at top-left. To have latitude increase upwards,
    # we set bottom=lat_min, top=lat_max. However, the reference image has
    # latitude decreasing downwards (26N at top, 32N at bottom).
    # So we use [lon_min, lon_max, lat_max, lat_min] to match the reference.
    plot_extent = [lon_min, lon_max, lat_max, lat_min]

    # Define tick positions and formatter
    lon_ticks = np.linspace(lon_min, lon_max, 7) # 122, 123, ..., 128
    lat_ticks = np.linspace(lat_min, lat_max, 7) # 26, 27, ..., 32
    lon_formatter = mticker.FormatStrFormatter('%.1f°E')
    lat_formatter = mticker.FormatStrFormatter('%.1f°N')

    # Determine shared color limits for true and predicted heatmaps
    vmin = min(true_data.min(), pred_data.min())
    vmax = max(true_data.max(), pred_data.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Plotting function for each heatmap
    def plot_single_heatmap(ax, data, title, cmap, norm, cbar_label):
        im = ax.imshow(data[:, :, 0], cmap=cmap, norm=norm, extent=plot_extent, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks) # Ticks correspond to values due to extent
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label)
        return im

    # Plot True Values
    plot_single_heatmap(axes[0], true_grid, f'True {feature_name}', 'viridis', norm, feature_name) # Changed cmap to viridis like reference

    # Plot Predicted Values
    plot_single_heatmap(axes[1], pred_grid, f'Predicted {feature_name}', 'viridis', norm, feature_name) # Changed cmap to viridis like reference

    # Plot Absolute Error
    error_norm = colors.Normalize(vmin=error_data.min(), vmax=error_data.max())
    plot_single_heatmap(axes[2], error_grid, f'Absolute Error', 'viridis', error_norm, 'Error Value') # Changed cmap to viridis

    # Overall title
    plt.suptitle(f'Sample {sample_idx+1}, Time Step {time_idx+1}: {feature_name} Comparison', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap image to: {save_path}")
    else:
        plt.show() # Show plot if not saving

    plt.close(fig) # Close the figure object to free memory

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
            save_path = os.path.join(output_dir, f"{feature_name}_sample{sample_idx}_month{t+1}.png")
        
        visualize_heatmaps(
            y_true, y_pred, sample_idx, t, feature_idx, 
            f"{feature_name} (月份 {t+1})", grid_size, save_path
        )
        
    return [os.path.join(output_dir, f"{feature_name}_sample{sample_idx}_month{t+1}.png") for t in range(n_timesteps)]

def evaluate_model(y_true, y_pred, feature_names=['Sea Temperature', 'Sea Salinity']):
    """
    评估模型性能，计算各种指标
    """
    results = {}
    
    # 转换为numpy数组
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # 整体评估
    mae_all = np.mean(np.abs(y_true_np - y_pred_np))
    rmse_all = np.sqrt(np.mean(np.square(y_true_np - y_pred_np)))
    mape_all = np.mean(np.abs((y_true_np - y_pred_np) / (y_true_np + 1e-5))) * 100
    wmape_all = np.sum(np.abs(y_true_np - y_pred_np)) / np.sum(np.abs(y_true_np)) * 100
    
    results['overall'] = {
        'MAE': mae_all,
        'RMSE': rmse_all,
        'MAPE': mape_all,
        'WMAPE': wmape_all
    }
    
    # 按特征评估
    for i, feature in enumerate(feature_names):
        mae = np.mean(np.abs(y_true_np[..., i] - y_pred_np[..., i]))
        rmse = np.sqrt(np.mean(np.square(y_true_np[..., i] - y_pred_np[..., i])))
        mape = np.mean(np.abs((y_true_np[..., i] - y_pred_np[..., i]) / (y_true_np[..., i] + 1e-5))) * 100
        wmape = np.sum(np.abs(y_true_np[..., i] - y_pred_np[..., i])) / np.sum(np.abs(y_true_np[..., i])) * 100
        
        results[feature] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'WMAPE': wmape
        }
    
    # 按预测时长评估
    horizon_results = []
    for t in range(y_true_np.shape[1]):
        mae_t = np.mean(np.abs(y_true_np[:, t] - y_pred_np[:, t]))
        rmse_t = np.sqrt(np.mean(np.square(y_true_np[:, t] - y_pred_np[:, t])))
        mape_t = np.mean(np.abs((y_true_np[:, t] - y_pred_np[:, t]) / (y_true_np[:, t] + 1e-5))) * 100
        wmape_t = np.sum(np.abs(y_true_np[:, t] - y_pred_np[:, t])) / np.sum(np.abs(y_true_np[:, t])) * 100
        
        horizon_results.append({
            'horizon': t+1,
            'MAE': mae_t,
            'RMSE': rmse_t,
            'MAPE': mape_t,
            'WMAPE': wmape_t
        })
    
    results['horizon'] = horizon_results
    
    return results

def plot_horizon_metrics(results, save_path=None):
    """
    绘制不同预测时长的误差指标
    """
    horizons = [r['horizon'] for r in results['horizon']]
    mae_values = [r['MAE'] for r in results['horizon']]
    rmse_values = [r['RMSE'] for r in results['horizon']]
    mape_values = [r['MAPE'] for r in results['horizon']]
    wmape_values = [r['WMAPE'] for r in results['horizon']]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(horizons, mae_values, 'o-', label='MAE')
    plt.plot(horizons, rmse_values, 's-', label='RMSE')
    plt.plot(horizons, mape_values, '^-', label='MAPE (%)')
    plt.plot(horizons, wmape_values, 'd-', label='WMAPE (%)')
    
    plt.xlabel('Prediction Month')
    plt.ylabel('Error Value')
    plt.title('Error Metrics across Different Prediction Horizons')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction horizon error metrics plot to: {save_path}")
    
    plt.close()

def plot_time_series_comparison(y_true, y_pred, sample_indices, time_indices, feature_idx, feature_name, grid_size=24, save_path=None):
    """
    绘制时间序列对比图，显示不同时间点的预测结果
    """
    n_samples = len(sample_indices)
    n_timesteps = len(time_indices)
    
    fig, axes = plt.subplots(n_timesteps, 3, figsize=(18, 6 * n_timesteps))
    
    for i, t_idx in enumerate(time_indices):
        for s_idx in sample_indices:
            # 获取数据
            true_data = y_true[s_idx, t_idx, :, feature_idx].cpu().numpy()
            pred_data = y_pred[s_idx, t_idx, :, feature_idx].cpu().numpy()
            error_data = np.abs(true_data - pred_data)
            
            # 重塑为网格
            true_grid = reshape_to_grid(true_data, grid_size)
            pred_grid = reshape_to_grid(pred_data, grid_size)
            error_grid = reshape_to_grid(error_data, grid_size)
            
            # 颜色范围
            vmin = min(true_data.min(), pred_data.min())
            vmax = max(true_data.max(), pred_data.max())
            
            # 绘制
            ax_row = axes[i] if n_timesteps > 1 else axes
            
            im1 = ax_row[0].imshow(true_grid[:, :, 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax_row[0].set_title(f'True {feature_name} (Month {t_idx+1})')
            fig.colorbar(im1, ax=ax_row[0], shrink=0.8)
            
            im2 = ax_row[1].imshow(pred_grid[:, :, 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax_row[1].set_title(f'Predicted {feature_name} (Month {t_idx+1})')
            fig.colorbar(im2, ax=ax_row[1], shrink=0.8)
            
            im3 = ax_row[2].imshow(error_grid[:, :, 0], cmap='Reds')
            ax_row[2].set_title(f'Absolute Error (Month {t_idx+1})')
            fig.colorbar(im3, ax=ax_row[2], shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved time series comparison plot to: {save_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='AGCRN Model Testing and Visualization')
    parser.add_argument('--model_dir', type=str, default='./shortpredict/logs/2025-04-20-18-37-49-eastsea_monthly', 
                        help='Model directory containing parameters and model files')
    parser.add_argument('--data_dir', type=str, default='./shortpredict/val_data/eastsea_monthly', 
                        help='Data directory')
    parser.add_argument('--sample_idx', type=int, default=0, 
                        help='Sample index to visualize')
    parser.add_argument('--feature', type=str, choices=['temperature', 'salinity', 'both'], default='both', 
                        help='Feature to visualize')
    parser.add_argument('--output_dir', type=str, default='./analysis_plots/visualization_results', 
                        help='Output directory')
    parser.add_argument('--animation', action='store_true', 
                        help='Create animation frames')
    parser.add_argument('--grid_size', type=int, default=24, 
                        help='Grid size')
    parser.add_argument('--input_len', type=int, default=12, 
                        help='Input sequence length')
    parser.add_argument('--output_len', type=int, default=6, 
                        help='Output sequence length')
    
    args = parser.parse_args()
    
    # 构建绝对路径
    current_dir = Path(__file__).parent.parent
    model_dir = os.path.join(current_dir, args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
    data_dir = os.path.join(current_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_dir = os.path.join(current_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载模型参数
    param_files = [f for f in os.listdir(model_dir) if f.startswith('params') and f.endswith('.json')]
    if not param_files:
        raise FileNotFoundError("Parameter file not found in the specified directory")
    
    with open(os.path.join(model_dir, param_files[0]), 'r') as f:
        params = json.load(f)
    
    # 确保使用正确的输入/输出长度
    params['input_len'] = args.input_len
    params['output_len'] = args.output_len
    
    print(f"Using configuration: input_len={params['input_len']}, output_len={params['output_len']}")
    
    # 找到最佳模型文件
    model_file = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError("best_model.pth not found in the specified directory")
    
    # 加载数据
    processed_data, test_index, scaler = load_data(
        data_dir, 
        input_len=params['input_len'], 
        output_len=params['output_len']
    )
    
    # 准备测试数据
    x_test, y_test = prepare_test_data(
        processed_data, 
        test_index, 
        params['input_len'], 
        params['output_len']
    )
    
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
    
    # 评估模型性能
    results = evaluate_model(y_test, y_pred, feature_names=['Sea Temperature', 'Sea Salinity'])
    
    # 打印评估结果
    print("\n===== Model Performance Evaluation =====")
    print("Overall Performance:")
    for metric, value in results['overall'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nSea Temperature Prediction Performance:")
    for metric, value in results['Sea Temperature'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nSea Salinity Prediction Performance:")
    for metric, value in results['Sea Salinity'].items():
        print(f"  {metric}: {value:.4f}")
    
    # 绘制预测时长误差图
    plot_horizon_metrics(results, os.path.join(output_dir, 'prediction_horizon_error.png'))
    
    # 开始可视化
    sample_idx = min(args.sample_idx, len(test_index) - 1)
    feature_map = {'temperature': 0, 'salinity': 1}
    
    if args.animation:
        # 创建动画帧
        if args.feature == 'both' or args.feature == 'temperature':
            create_animation_frames(
                y_test, y_pred, sample_idx, 0, 'Sea Temperature', 
                args.grid_size, os.path.join(output_dir, 'temperature_frames')
            )
        
        if args.feature == 'both' or args.feature == 'salinity':
            create_animation_frames(
                y_test, y_pred, sample_idx, 1, 'Sea Salinity', 
                args.grid_size, os.path.join(output_dir, 'salinity_frames')
            )
    else:
        # 可视化特定时间步
        time_indices = [0, params['output_len']//2, params['output_len']-1]  # 开始、中间、结束
        
        # 创建单个热力图
        mid_timestep = params['output_len'] // 2
        
        if args.feature == 'both' or args.feature == 'temperature':
            visualize_heatmaps(
                y_test, y_pred, sample_idx, mid_timestep, 0, 'Sea Temperature', 
                args.grid_size, os.path.join(output_dir, f'temperature_month{mid_timestep+1}_sample{sample_idx}.png')
            )
            
            # 创建时间序列对比图
            plot_time_series_comparison(
                y_test, y_pred, [sample_idx], time_indices, 0, 'Sea Temperature',
                args.grid_size, os.path.join(output_dir, 'temperature_time_series.png')
            )
        
        if args.feature == 'both' or args.feature == 'salinity':
            visualize_heatmaps(
                y_test, y_pred, sample_idx, mid_timestep, 1, 'Sea Salinity', 
                args.grid_size, os.path.join(output_dir, f'salinity_month{mid_timestep+1}_sample{sample_idx}.png')
            )
            
            # 创建时间序列对比图
            plot_time_series_comparison(
                y_test, y_pred, [sample_idx], time_indices, 1, 'Sea Salinity',
                args.grid_size, os.path.join(output_dir, 'salinity_time_series.png')
            )
    
    print(f"\nVisualization results saved to: {output_dir}")
    
    # 保存评估结果
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        # 转换numpy数组为Python列表
        json_results = {
            'overall': {k: float(v) for k, v in results['overall'].items()},
            'Sea Temperature': {k: float(v) for k, v in results['Sea Temperature'].items()},
            'Sea Salinity': {k: float(v) for k, v in results['Sea Salinity'].items()},
            'horizon': [
                {k: float(v) if k != 'horizon' else v for k, v in h.items()}
                for h in results['horizon']
            ]
        }
        json.dump(json_results, f, indent=2)

if __name__ == "__main__":
    main() 