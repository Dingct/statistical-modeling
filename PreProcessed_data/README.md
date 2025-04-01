
预处理好的数据
使用np读入即可，示例：yangtze_data = np.load('yangtze_processed.npy') 

yangtze_processed.npy是(3600,64,2)的三维数组，第三维是[anc , smap] 

eastsea_processed.npy是(3600,24*24,2)的三维数组，第三维是[anc , smap] 

数据使用了克里金 (Krige) 的时空插值，专门针对时空地理数据插值后进行了标准化，68% 的数据点会落在均值 ±1 个标准差范围内，95% 的数据落在 ±2 个标准差内，99.7% 的数据落在 ±3 个标准差内（68-95-99.7 法则）。 

其余两张图片是插值前后的可视化
