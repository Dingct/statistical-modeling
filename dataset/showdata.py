import netCDF4 as nc  

# 文件路径  
file_path = 'lab/dataset/eastsea.nc'

# 打开 NetCDF 文件  
dataset = nc.Dataset(file_path)  

# 打印每个变量的特征  
print("\n变量特征:")  
for var in dataset.variables:  
    variable = dataset.variables[var]  
    # 获取变量数据的形状  
    shape = variable.shape  
    print(f"{var}: shape={shape}")  

# 打印坐标维度信息  
print("\n维度信息:")  
for dim in dataset.dimensions:  
    print(f"{dim}: {dataset.dimensions[dim].size}")  

# 最后关闭文件  
dataset.close()  