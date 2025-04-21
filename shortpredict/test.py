import util
import argparse
import torch
from model import Ding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util import StandardScaler
import os

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", help="") # 若有GPU换
parser.add_argument("--data", type=str, default="eastsea", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim") # 海盐，海温，日期
parser.add_argument("--channels", type=int, default=32, help="dimension of nodes")
parser.add_argument("--num_nodes", type=int, default=24*24, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="output_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument('--checkpoint', type=str,
                    default='./logs/2025-04-13-12-03-12-eastsea', help='') # 换对应的路径
parser.add_argument('--plotheatmap', type=str, default='True', help='')
args = parser.parse_args()

def test(scaler,yhat,realy,dimen,args):
    amae = []
    amape = []
    awmape = []
    armse = []
    
    for i in range(args.output_len):
        # pred = scaler.inverse_transform(yhat[:, i, :, :])
        # real = scaler.inverse_transform(realy[:, i, :, :])
        pred = yhat[:, i, :, :]
        real = realy[:, i, :, :]
        if(dimen=="all"):
            metrics = util.testmetrics(pred, real)
        else:
            metrics = util.testmetric(pred, real)
        log = 'Evaluate best model on {} for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}'
        print(log.format(dimen, i+1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])


    log = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}'
    print(log.format(args.output_len,np.mean(amae), np.mean(amape), np.mean(armse),np.mean(awmape)))
    # print(realy)
    # print(yhat)
    realy = scaler.inverse_transform(realy)
    realy = realy.to("cpu")
    yhat = scaler.inverse_transform(yhat)
    yhat = yhat.to("cpu")
    # print(realy)
    # print(yhat)
    print(realy.shape)
    print(yhat.shape)

    torch.save(realy,f"{args.checkpoint}/{dimen}_real.pt")
    torch.save(yhat,f"{args.checkpoint}/{dimen}_pred.pt")

def main():
    
    # 目前只用2个数据集，input_len=12,output_len=12
    if args.data == "eastsea":
        args.data = "./val_data/" + args.data
        args.num_nodes = 24*24
    elif args.data == "Yangtze":
        args.data = "./val_data/" + args.data
        args.num_nodes = 8*8

    device = torch.device(args.device)

    # 要改load哪个模型
    model = Ding(
            device, args.input_dim, args.channels, args.num_nodes, args.input_len, args.output_len, args.dropout
        )
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint,"best_model.pth")))
    model.eval()

    print('model load successfully')

    train_loader, valid_loader, test_loader, scaler= util.load_data_with_dataloader(
        args.data, args.batch_size, args.input_len, args.output_len, args.device, args.batch_size, args.batch_size
    )

    outputs = []
    test_labels = []

    for iter, (x, y) in enumerate(test_loader):
        testx = x.to(device)
        testy = y.to(device)
        with torch.no_grad():
            preds = model(testx)         # [64,1,576,12]
        outputs.append(preds.squeeze())   # [64,576,12]
        test_labels.append(testy[...,0:2])

    realy = torch.cat(test_labels, dim=0) # B 12 576 2
    yhat = torch.cat(outputs, dim=0)  # B 12 576 2
    yhat = yhat[:realy.size(0), ...]
    
    scalers = StandardScaler(mean=scaler.mean[0], std=scaler.std[0])
    scalert = StandardScaler(mean=scaler.mean[1], std=scaler.std[1])
    test(scalers,yhat[...,0:1],realy[...,0:1],"salt",args)
    test(scalert,yhat[...,1:2],realy[...,1:2],"temperature",args)
    test(scaler,yhat,realy,"all",args)

if __name__ == "__main__":
    main()
