import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class TrafficFlowDataset(Dataset):
    def __init__(self, data, index, dataset_type):
        self.dataset_data = []
        self.dataset_labels = []
        for (start, end, label_end) in index[dataset_type]:
            inputs = data[start:end, :, :]
            labels = data[end:label_end, :, :]
            self.dataset_data.append(inputs)
            self.dataset_labels.append(labels)

    def __len__(self):
        return len(self.dataset_data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.dataset_data[idx]).float(), torch.from_numpy(self.dataset_labels[idx]).float()
        return sample
    
def load_data_with_dataloader(dataset_dir, batch_size, input_len, output_len, device, valid_batch_size=None, test_batch_size=None):
    with open(os.path.join(dataset_dir, "data_in_{0}_out_{1}.pkl").format(input_len, output_len), "rb") as f:
        data_file = pickle.load(f)
    with open(os.path.join(dataset_dir, "index_in_{0}_out_{1}.pkl").format(input_len, output_len), "rb") as f:
        index = pickle.load(f)
    with open(os.path.join(dataset_dir, "scaler_in_{0}_out_{1}.pkl").format(input_len, output_len), "rb") as f:
        scaler_ = pickle.load(f)
    
    # 获取平均值和标准差
    mean, std = scaler_["args"]["mean"], scaler_["args"]["std"]
    mean = torch.tensor(mean, dtype=torch.float32).to(device)  
    std = torch.tensor(std, dtype=torch.float32).to(device)  
    scaler = StandardScaler(mean=mean, std=std)

    data_processed = data_file["processed_data"]
    
    # 实例化数据集
    train_dataset = TrafficFlowDataset(data_processed, index, 'train')
    valid_dataset = TrafficFlowDataset(data_processed, index, 'valid')
    test_dataset = TrafficFlowDataset(data_processed, index, 'test')
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size or batch_size, shuffle=False,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size or batch_size, shuffle=False,num_workers=4)
    
    return train_loader, valid_loader, test_loader, scaler

def MAE_torch(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(prediction-target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean squared error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean squared error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (prediction-target)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def RMSE_torch(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """root mean squared error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.

    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))


def MAPE_torch(prediction: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    target = torch.where(torch.abs(target) < 1e-4, torch.zeros_like(target), target)
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(prediction-target)/target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def WMAPE_torch(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked weighted absolute percentage error (WAPE)

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    prediction, target = prediction * mask, target * mask
    loss =  torch.sum(torch.abs(prediction-target)) / (torch.sum(torch.abs(target)) + 5e-5)
    return torch.mean(loss)

    
def metric(pred, real):
    mae = MAE_torch(pred, real, 0.0).item()
    mape = MAPE_torch(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    rmse = RMSE_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape

# 多因素
def metrics(predict, real):
    loss1 = MAE_torch(predict[...,:1], real[...,:1], 0.0)
    loss2 = MAE_torch(predict[...,1:2], real[...,1:2], 0.0)
    mape1 = MAPE_torch(predict[...,:1], real[...,:1], 0.0).item()
    mape2 = MAPE_torch(predict[...,1:2], real[...,1:2], 0.0).item()
    rmse1 = RMSE_torch(predict[...,:1], real[...,:1], 0.0).item()
    rmse2 = RMSE_torch(predict[...,1:2], real[...,1:2], 0.0).item()
    wmape1 = WMAPE_torch(predict[...,:1], real[...,:1], 0.0).item()
    wmape2 = WMAPE_torch(predict[...,1:2], real[...,1:2], 0.0).item()

    loss = (loss1+loss2)/2 # 对loss取平均值
    mape = (mape1+mape2)/2
    rmse = (rmse1+rmse2)/2
    wmape = (wmape1+wmape2)/2

    return loss, mape, rmse, wmape

def testmetrics(predict, real):
    loss1 = MAE_torch(predict[...,:1], real[...,:1], 0.0).item()
    loss2 = MAE_torch(predict[...,1:2], real[...,1:2], 0.0).item()
    mape1 = MAPE_torch(predict[...,:1], real[...,:1], 0.0).item()
    mape2 = MAPE_torch(predict[...,1:2], real[...,1:2], 0.0).item()
    rmse1 = RMSE_torch(predict[...,:1], real[...,:1], 0.0).item()
    rmse2 = RMSE_torch(predict[...,1:2], real[...,1:2], 0.0).item()
    wmape1 = WMAPE_torch(predict[...,:1], real[...,:1], 0.0).item()
    wmape2 = WMAPE_torch(predict[...,1:2], real[...,1:2], 0.0).item()

    loss = (loss1+loss2)/2 # 对loss取平均值
    mape = (mape1+mape2)/2
    rmse = (rmse1+rmse2)/2
    wmape = (wmape1+wmape2)/2

    return loss, mape, rmse, wmape
