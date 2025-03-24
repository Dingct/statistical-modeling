# 数据集
<h1 align="center"><ins>S</ins>patio-<ins>T</ins>emporal <ins>G</ins>raph <ins>C</ins>onvolutional <ins>N</ins>etworks: <br> A Deep Learning Framework for Traffic Forecasting</h1>
<p align="center">
    <a href="https://www.ijcai.org/proceedings/2018/0505.pdf"><img src="https://img.shields.io/badge/-Paper-grey?logo=read%20the%20docs&logoColor=green" alt="Paper"></a>
    <a href="https://github.com/VeritasYin/STGCN_IJCAI-18"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-BSD%202--Clause-red.svg"></a>
    <a href="https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.stgcn.STConv"><img src="https://img.shields.io/badge/PyG_Temporal-STConv-blue" alt=PyG_Temporal"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVeritasYin%2FSTGCN_IJCAI-18&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false"/></a>
</p>
纬度0.25度*经度0.25度为一格，共3600天数据
## Yangtze.nc
长江口数据，经纬度范围：'lat_range': (30, 32), 'lon_range': (120, 123)

变量特征:

时间 time: shape=(3600,)
纬度 latitude: shape=(8,)
经度 longitude: shape=(12,)
海盐 smap_sss: shape=(3600, 8, 12)
海温 anc_sst: shape=(3600, 8, 12)
风速 smap_spd: shape=(3600, 8, 12)
海盐不确定度 smap_sss_uncertainty: shape=(3600, 8, 12)

维度信息:

时间 time: 3600
纬度 latitude: 8
经度 longitude: 12
## eastsea.nc
东海数据，经纬度范围：'lat_range': (25, 35), 'lon_range': (120, 130)

变量特征:

时间 time: shape=(3600,)
纬度 latitude: shape=(40,)
经度 longitude: shape=(40,)
海盐 smap_sss: shape=(3600, 40, 40)
海温 anc_sst: shape=(3600, 40, 40)
风速 smap_spd: shape=(3600, 40, 40)
海盐不确定度 smap_sss_uncertainty: shape=(3600, 40, 40)

维度信息:

time: 3600
latitude: 40
longitude: 40
## showdata.py
用于展示数据
