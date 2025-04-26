import numpy as np
import pandas as pd
import argparse
import time
import os
from util import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.vector_ar.var_model import VAR
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="ha", help="model type") # 'ha', 'var', 'svr'
    parser.add_argument("--data", type=str, default="eastsea", help="data path")
    parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
    parser.add_argument("--num_nodes", type=int, default=24*24, help="number of nodes")
    parser.add_argument("--input_len", type=int, default=12, help="input_len")
    parser.add_argument("--output_len", type=int, default=12, help="out_len")
    parser.add_argument("--save", type=str, default="./logs_ml/", help="save path")
    return parser.parse_args()

def load_data(data_path, input_len, output_len):
    """加载数据并转换为适合传统机器学习模型的格式"""
    with open(os.path.join(data_path, f"data_in_{input_len}_out_{output_len}.pkl"), "rb") as f:
        data_file = pickle.load(f)
    with open(os.path.join(data_path, f"index_in_{input_len}_out_{output_len}.pkl"), "rb") as f:
        index = pickle.load(f)
    with open(os.path.join(data_path, f"scaler_in_{input_len}_out_{output_len}.pkl"), "rb") as f:
        scaler_ = pickle.load(f)
    
    # 获取处理后的数据
    data = data_file["processed_data"][...,0:2]
    scaler = StandardScaler(mean=scaler_["args"]["mean"], std=scaler_["args"]["std"])
    return data, index, scaler

def reshape_data(data, index, dataset_type):
    """将数据重塑为适合时序预测的格式"""
    X, y = [], []
    for (start, end, label_end) in index[dataset_type]:
        # 输入数据 [T, N, C]
        inputs = data[start:end, :, :]
        # 目标数据 [T, N, C]
        labels = data[end:label_end, :, :]
        
        # 重塑为 [N, T, C]
        inputs = inputs.transpose(1, 0, 2)
        labels = labels.transpose(1, 0, 2)
        
        X.append(inputs)
        y.append(labels)
    
    # 合并所有样本 [样本数*N, T, C]
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

class HistoricalAverage:
    """历史平均模型"""
    def __init__(self):
        self.history = None
        
    def fit(self, X, y):
        """存储历史数据"""
        self.history = X
        
    def predict(self, X):
        """使用历史平均值进行预测"""
        if self.history is None:
            raise ValueError("Model not fitted yet")
            
        # X: [样本数*N, T, C]
        # 计算历史平均值
        hist_mean = np.mean(X, axis=1, keepdims=True)  # [样本数*N, 1, C]
        # 预测未来12步
        predictions = np.tile(hist_mean, (1, 12, 1))  # [样本数*N, 12, C]
        return predictions

class VectorAutoRegression:
    """向量自回归模型"""
    def __init__(self, maxlags=2):
        self.maxlags = maxlags
        
    def fit(self, X, y):
        """VAR模型不需要训练"""
        pass
            
    def predict(self, X):
        """使用VAR模型进行预测"""
        # X: [样本数*N, T, C]
        predictions = []
        
        # 对每个样本进行预测
        for i in tqdm(range(X.shape[0]), desc="VAR预测进度"):
            # 获取当前样本的数据 [T, C]
            sample_data = X[i]
            
            # 确保数据维度正确
            if sample_data.shape[0] < self.maxlags:
                raise ValueError(f"输入序列长度({sample_data.shape[0]})小于maxlags({self.maxlags})")
            
            # 创建VAR模型并进行预测
            model = VAR(sample_data)
            fitted_model = model.fit(maxlags=self.maxlags)
            pred = fitted_model.forecast(sample_data, steps=12)  # [12, C]
            predictions.append(pred)
        
        # 将预测结果合并为[样本数*N, 12, C]
        predictions = np.stack(predictions, axis=0)
        return predictions

class SVRWrapper:
    """支持向量回归模型包装器"""
    def __init__(self):
        self.models = None  # 存储训练好的模型
        
    def fit(self, X, y):
        """训练SVR模型"""
        # X: [样本数*N, T, C]
        # y: [样本数*N, T, C]
        
        # 为每个特征创建一个SVR模型
        self.models = []
        for c in range(2):  # 两个特征
            # 准备训练数据
            # 将所有样本的特征数据展平
            X_train = X[:, :, c].reshape(-1, 1)  # [样本数*N*T, 1]
            y_train = y[:, :, c].reshape(-1)  # [样本数*N*T]
            
            # 创建并训练SVR模型
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(X_train, y_train)
            self.models.append(model)
                
    def predict(self, X):
        """使用训练好的SVR模型进行预测"""
        if self.models is None:
            raise ValueError("Model not fitted yet")
            
        # X: [样本数*N, T, C]
        predictions = []
        
        # 对每个样本进行预测
        for i in tqdm(range(X.shape[0]), desc="SVR预测进度"):
            # 获取当前样本的数据 [T, C]
            sample_data = X[i]
            
            # 使用训练好的模型进行预测
            preds = []
            for c in range(2):  # 两个特征
                # 准备预测数据
                X_pred = sample_data[:, c].reshape(-1, 1)  # [T, 1]
                
                # 进行预测
                pred = self.models[c].predict(X_pred)  # [T]
                preds.append(pred)
            
            # 合并两个特征的预测结果 [T, C]
            pred = np.stack(preds, axis=-1)
            predictions.append(pred)
        
        # 将预测结果合并为[样本数*N, T, C]
        predictions = np.stack(predictions, axis=0)
        return predictions

def create_model(model_type):
    """创建模型"""
    if model_type == 'ha':
        return HistoricalAverage()
    elif model_type == 'var':
        return VectorAutoRegression(maxlags=2)
    elif model_type == 'svr':
        return SVRWrapper()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate(y_true, y_pred):
    """评估模型性能"""
    # y_true, y_pred: [样本数*N, T, C]
    
    # 分别计算两个特征维度的指标
    # print(y_true[0])
    # print(y_pred[0])
    mae1 = mean_absolute_error(y_true[:, :, 0], y_pred[:, :, 0])
    mae2 = mean_absolute_error(y_true[:, :, 1], y_pred[:, :, 1])
    
    mape1 = np.mean(np.abs((y_true[:, :, 0] - y_pred[:, :, 0]) / (y_true[:, :, 0] + 1e-5))) * 100
    mape2 = np.mean(np.abs((y_true[:, :, 1] - y_pred[:, :, 1]) / (y_true[:, :, 1] + 1e-5))) * 100
    
    rmse1 = np.sqrt(mean_squared_error(y_true[:, :, 0], y_pred[:, :, 0]))
    rmse2 = np.sqrt(mean_squared_error(y_true[:, :, 1], y_pred[:, :, 1]))
    
    wmape1 = np.sum(np.abs(y_true[:, :, 0] - y_pred[:, :, 0])) / (np.sum(np.abs(y_true[:, :, 0])) + 1e-5) * 100
    wmape2 = np.sum(np.abs(y_true[:, :, 1] - y_pred[:, :, 1])) / (np.sum(np.abs(y_true[:, :, 1])) + 1e-5) * 100
    
    # 对两个特征维度的指标取平均
    mae = (mae1 + mae2) / 2
    mape = (mape1 + mape2) / 2
    rmse = (rmse1 + rmse2) / 2
    wmape = (wmape1 + wmape2) / 2
    
    return mae1, mape1, rmse1, wmape1, mae2, mape2, rmse2, wmape2, mae, mape, rmse, wmape

def main():
    args = parse_args()
    args.data = "./val_data/" + args.data
    print(args)
    
    # 创建保存目录
    save_path = os.path.join(args.save, args.data)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 加载数据
    data, index, scaler = load_data(args.data, args.input_len, args.output_len)
    
    # 准备训练数据
    X_train, y_train = reshape_data(data, index, 'train')
    X_val, y_val = reshape_data(data, index, 'valid')
    X_test, y_test = reshape_data(data, index, 'test')
    X_train = X_train[:500]
    y_train = y_train[:500]
    # X_test = X_test[:1000]
    # y_test = y_test[:1000]
    
    # 创建模型
    model = create_model(args.model_type)
    
    # 训练模型
    print("开始训练...")
    start_time = time.time()
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 评估模型
    print("\n评估结果:")
    
    for dataset_name, X, y in [("测试集", X_test, y_test)]:
        y_pred = model.predict(X)
        # 重塑为 [N, T, C]
        y = y.reshape(-1, 12, 2)
        y_pred = y_pred.reshape(-1, 12, 2)
        print(X.shape, y.shape, y_pred.shape)
        mae1, mape1, rmse1, wmape1, mae2, mape2, rmse2, wmape2, mae, mape, rmse, wmape = evaluate(y, y_pred)
        print(f"{dataset_name} - MAE1: {mae1:.4f}, MAPE1: {mape1:.4f}%, RMSE1: {rmse1:.4f}, WMAPE1: {wmape1:.4f}%")
        print(f"{dataset_name} - MAE2: {mae2:.4f}, MAPE2: {mape2:.4f}%, RMSE2: {rmse2:.4f}, WMAPE2: {wmape2:.4f}%")
        print(f"{dataset_name} - MAE: {mae:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.4f}, WMAPE: {wmape:.4f}%")
    

    # 保存模型
    model_path = os.path.join(save_path, f"{args.model_type}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model
        }, f)
    print(f"\n模型已保存到: {model_path}")

if __name__ == "__main__":
    main() 