# 长江口、东海，海温、海盐数据集
纬度0.25度*经度0.25度为一格，共3600天数据
## Yangtze.nc
长江口数据，经纬度范围：'lat_range': (30, 32), 'lon_range': (121, 123)

变量特征:

时间 time: shape=(3600,)
纬度 latitude: shape=(8,)
经度 longitude: shape=(8,)
海盐 smap_sss: shape=(3600, 8, 8)
海温 anc_sst: shape=(3600, 8, 8)
风速 smap_spd: shape=(3600, 8, 8)
海盐不确定度 smap_sss_uncertainty: shape=(3600, 8, 8)

维度信息:

时间 time: 3600
纬度 latitude: 8
经度 longitude: 8
## eastsea.nc
东海数据，经纬度范围：'lat_range': (26, 32), 'lon_range': (122, 128)

变量特征:

时间 time: shape=(3600,)
纬度 latitude: shape=(24,)
经度 longitude: shape=(24,)
海盐 smap_sss: shape=(3600, 24, 24)
海温 anc_sst: shape=(3600, 24, 24)
风速 smap_spd: shape=(3600, 24, 24)
海盐不确定度 smap_sss_uncertainty: shape=(3600, 24, 24)

维度信息:

time: 3600
latitude: 24
longitude: 24
## showdata.py
用于展示数据
