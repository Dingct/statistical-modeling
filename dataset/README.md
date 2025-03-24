# 长江口、东海海温、海盐数据集
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
