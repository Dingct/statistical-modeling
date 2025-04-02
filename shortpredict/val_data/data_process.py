import os
import torch
import pickle
import argparse
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta


def standard_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int, norm_each_channel: int = False) -> np.array:
    """
    标准化数据。

    参数:
        data (np.array): 原始时间序列数据。
        output_dir (str): 输出目录路径。
        train_index (list): 训练数据的索引。
        history_seq_len (int): 历史序列长度。
        future_seq_len (int): 未来序列长度。
        norm_each_channel (bool): 是否对每个通道进行单独归一化。

    返回:
        np.array: 归一化后的时间序列数据。
    """

    # 获取训练数据
    data_train = data[:train_index[-1][1], ...]

    if norm_each_channel:
        # 计算每个通道的均值和标准差 这里mean是[2]维的
        mean = data_train.mean(axis=(0, 1))  # 沿时间和节点维度计算，得到每个通道的均值
        std = data_train.std(axis=(0, 1))
    else:
        # 全局均值和标准差，所有通道一起计算
        mean = data_train.mean()
        std = data_train.std()

    print("训练数据的均值:", mean)
    print("训练数据的标准差:", std)

    # 保存归一化参数
    scaler = {}
    scaler["func"] = re_standard_transform.__name__
    scaler["args"] = {"mean": mean, "std": std}
    
    # 保存归一化参数到文件
    with open(output_dir + "/scaler_in_{0}_out_{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(scaler, f)

    # 定义归一化函数
    def normalize(x):
        return (x - mean) / std

    # 归一化数据
    data_norm = normalize(data)
    return data_norm


def re_standard_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard re-transformation.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """

    mean, std = kwargs["mean"], kwargs["std"]
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).type_as(data).to(data.device).view(1, 1, -1)
        std = torch.from_numpy(std).type_as(data).to(data.device).view(1, 1, -1)
    data = data * std
    data = data + mean
    return data


def generate_data(args: argparse.Namespace):
    """
    预处理并生成训练/验证/测试数据集。

    参数:
        args (argparse.Namespace): 预处理的配置参数
    """

    # 提取参数配置
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    norm_each_channel = args.norm_each_channel

    # 读取数据
    data = nc.Dataset(data_file_path, "r") # 前2维分别是海温和海盐
    data = data.variables["data"][:]
    print("原始时间序列形状: {0}".format(data.shape))

    # 划分数据集
    l, n, f= data.shape # 形状为 (时间步, 节点数, 2)
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("训练样本数量: {0}".format(train_num))
    print("验证样本数量: {0}".format(valid_num))
    print("测试样本数量: {0}".format(test_num))

    # 生成索引列表
    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t - history_seq_len, t, t + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num + valid_num: train_num + valid_num + test_num]

    print(len(train_index))
    # 归一化数据
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)
    feature_list = [data_norm]
    print("data_shape",data_norm.shape)

    # 添加时间特征，day_in_year
    start_date = datetime(2015, 5, 4)
    time_features = []
    for i in range(data_norm.shape[0]):
        current_date = start_date + timedelta(days=i) 
        day_in_year_feature = (current_date - datetime(current_date.year, 1, 1)).days / 365 - 0.5  
        time_features.append([day_in_year_feature])
    time_features = np.array(time_features)[:, np.newaxis, :]
    feature_list.append(time_features.repeat(data_norm.shape[1], axis=1))

    processed_data = np.concatenate(feature_list, axis=-1)
    print("processed_data_shape",processed_data.shape)
    
    # 保存数据和索引
    index = {"train": train_index, "valid": valid_index, "test": test_index}
    with open(f"{output_dir}/index_in_{history_seq_len}_out_{future_seq_len}.pkl", "wb") as f:
        pickle.dump(index, f)

    data = {"processed_data": processed_data}
    with open(f"{output_dir}/data_in_{history_seq_len}_out_{future_seq_len}.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # 输入为nc文件，形状为[L,R*R,2] # L: 时间步，R: 经纬度网格数，2: 海温、海盐
    # 窗口大小用于生成历史序列和目标序列
    data_list = ['eastsea','Yangtze']

    for data in data_list:
        HISTORY_SEQ_LEN = 12
        FUTURE_SEQ_LEN = 12

        TRAIN_RATIO = 0.6
        VALID_RATIO = 0.2

        DATASET_NAME = data

        OUTPUT_DIR = f"{DATASET_NAME}"
        DATA_FILE_PATH = f"{DATASET_NAME}.nc"

        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录。")
        parser.add_argument("--data_file_path", type=str, default=DATA_FILE_PATH, help="原始数据路径。")
        parser.add_argument("--history_seq_len", type=int, default=HISTORY_SEQ_LEN, help="序列长度。")
        parser.add_argument("--future_seq_len", type=int, default=FUTURE_SEQ_LEN, help="序列长度。")
        parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="训练比例")
        parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO, help="验证比例。")
        parser.add_argument("--norm_each_channel", type=bool,  default=True, help="归一化每个通道。") # 对每一个通道单独归一化
        args = parser.parse_args()

        # 打印参数
        print("-" * (20 + 45 + 5))
        for key, value in sorted(vars(args).items()):
            print("|{0:>20} = {1:<45}|".format(key, str(value)))
        print("-" * (20 + 45 + 5))

        # 创建输出目录
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        generate_data(args)
