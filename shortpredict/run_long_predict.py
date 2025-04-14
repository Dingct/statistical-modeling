#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import os
import sys
import json
import time
from datetime import datetime

# 预定义的训练配置
CONFIGS = {
    "short": {  # 短时预测：12小时预测12小时
        "input_len": 12,
        "output_len": 12,
        "batch_size": 64,
        "channels": 64,
        "num_layers": 2,
        "epochs": 300
    },
    "medium": {  # 中时预测：12小时预测24小时
        "input_len": 12,
        "output_len": 24,
        "batch_size": 32,
        "channels": 96,
        "num_layers": 2,
        "epochs": 400
    },
    "long": {  # 长时预测：24小时预测24小时
        "input_len": 24,
        "output_len": 24,
        "batch_size": 32,
        "channels": 128,
        "num_layers": 3,
        "epochs": 500
    },
    "very_long": {  # 超长时预测：24小时预测48小时
        "input_len": 24,
        "output_len": 48,
        "batch_size": 16,
        "channels": 160,
        "num_layers": 3,
        "epochs": 600
    }
}

def save_run_info(config, args, cmd):
    """保存运行信息到日志文件"""
    # 创建日志目录
    log_dir = "./logs/runs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建运行信息字典
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_info = {
        "timestamp": timestamp,
        "preset": args.preset,
        "model": "agcrn",
        "dataset": args.data,
        "device": args.device,
        "input_len": config["input_len"],
        "output_len": config["output_len"],
        "batch_size": config["batch_size"],
        "channels": config["channels"],
        "num_layers": config["num_layers"],
        "epochs": config["epochs"],
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "cheb_k": args.cheb_k,
        "embed_dim": args.embed_dim,
        "command": " ".join(cmd)
    }
    
    # 保存为JSON文件
    log_file = f"{log_dir}/run_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(run_info, f, indent=4)
    
    return log_file, timestamp

def main():
    parser = argparse.ArgumentParser(description="长时时空序列预测训练脚本")
    parser.add_argument("--preset", type=str, default="medium", 
                        choices=["short", "medium", "long", "very_long"],
                        help="预设配置，可选：short, medium, long, very_long")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="计算设备，例如'cpu'、'cuda:0'或'mps'(macOS)")
    parser.add_argument("--data", type=str, default="eastsea", 
                        help="数据集名称")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="丢弃率")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="学习率")
    parser.add_argument("--cheb_k", type=int, default=3, 
                        help="Chebyshev多项式阶数")
    parser.add_argument("--embed_dim", type=int, default=10, 
                        help="节点嵌入维度")
    
    args = parser.parse_args()
    
    # 获取预设配置
    config = CONFIGS[args.preset]
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--model", "agcrn",  # 使用AGCRN模型
        "--device", args.device,
        "--data", args.data,
        "--input_len", str(config["input_len"]),
        "--output_len", str(config["output_len"]),
        "--batch_size", str(config["batch_size"]),
        "--channels", str(config["channels"]),
        "--num_layers", str(config["num_layers"]),
        "--epochs", str(config["epochs"]),
        "--dropout", str(args.dropout),
        "--learning_rate", str(args.learning_rate),
        "--cheb_k", str(args.cheb_k),
        "--embed_dim", str(args.embed_dim)
    ]
    
    # 保存运行信息
    log_file, timestamp = save_run_info(config, args, cmd)
    
    # 打印训练配置
    print("=" * 50)
    print(f"运行长时预测训练 - 预设配置: {args.preset}")
    print(f"运行时间戳: {timestamp}")
    print(f"使用模型: AGCRN")
    print(f"数据集: {args.data}")
    print(f"设备: {args.device}")
    print(f"输入序列长度: {config['input_len']}")
    print(f"输出预测长度: {config['output_len']}")
    print(f"批量大小: {config['batch_size']}")
    print(f"通道数: {config['channels']}")
    print(f"模型层数: {config['num_layers']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"学习率: {args.learning_rate}")
    print(f"丢弃率: {args.dropout}")
    print(f"Chebyshev多项式阶数: {args.cheb_k}")
    print(f"节点嵌入维度: {args.embed_dim}")
    print(f"运行日志: {log_file}")
    print("=" * 50)
    
    # 记录运行开始时间
    start_time = time.time()
    
    # 执行训练命令
    try:
        subprocess.run(cmd)
        status = "完成"
    except Exception as e:
        status = f"出错: {str(e)}"
    
    # 记录运行结束时间和状态
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 更新运行状态
    with open(log_file, "r") as f:
        run_info = json.load(f)
    
    run_info["status"] = status
    run_info["elapsed_time"] = elapsed_time
    run_info["elapsed_time_formatted"] = f"{elapsed_time//3600:.0f}小时 {(elapsed_time%3600)//60:.0f}分钟 {elapsed_time%60:.0f}秒"
    
    with open(log_file, "w") as f:
        json.dump(run_info, f, indent=4)
    
    print("=" * 50)
    print(f"运行状态: {status}")
    print(f"运行时间: {run_info['elapsed_time_formatted']}")
    print(f"完整日志已保存: {log_file}")
    print("=" * 50)

if __name__ == "__main__":
    main() 