
预处理好的数据

读取示例：
```python
import netCDF4 as nc


dataset = nc.Dataset("yangtze.nc", "r")

print(dataset.variables.keys())
# 只有一个key就是'data'

# 读取数据
data = dataset.variables["data"][:]
print(data.shape)  # 输出: (3600, 64, 2)

# 关闭文件
dataset.close()

```

yangtze_processed.nc的dataset.variables["data"]是(3600,64,2)的三维数组，第三维是[anc , smap] 

eastsea_processed.npy的dataset.variables["data"]是(3600,24*24,2)的三维数组，第三维是[anc , smap] 

数据使用了克里金 (Krige) 的时空插值，专门针对时空地理数据

没有标准化等任何其他额外操作 

其余两张图片是插值前后的可视化
