import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm
import netCDF4 as nc


smap_sss = np.load('smap_sss.npy')
anc_sst = np.load('anc_sst.npy')
time = np.load('time.npy')
lat = np.load('latitude.npy')
lon = np.load('longitude.npy')

def spatiotemporal_kriging_interpolation(missed_data, time, lat, lon):

    filled_data = np.copy(missed_data)

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    for t in tqdm(range(len(time)), desc="Processing time steps"):
        time_slice = missed_data[t, :, :]

        if np.isnan(time_slice).any():
            valid_mask = ~np.isnan(time_slice)
            valid_lats = lat_grid[valid_mask]
            valid_lons = lon_grid[valid_mask]
            valid_values = time_slice[valid_mask]

            nan_mask = np.isnan(time_slice)
            nan_lats = lat_grid[nan_mask]
            nan_lons = lon_grid[nan_mask]

            if len(valid_values) >= 3:
                try:

                    OK = OrdinaryKriging(
                        valid_lons,
                        valid_lats,
                        valid_values,
                        variogram_model='spherical',
                        verbose=False,
                        enable_plotting=False
                    )


                    interpolated_values, _ = OK.execute('points', nan_lons, nan_lats)


                    filled_data[t][nan_mask] = interpolated_values
                except Exception as e:
                    print(f"克里金插值在时间步 {t} 失败: {str(e)}")
                    print("使用临近插值代替...")

                    interpolated_values = griddata(
                        np.column_stack((valid_lons, valid_lats)),
                        valid_values,
                        np.column_stack((nan_lons, nan_lats)),
                        method='nearest'
                    )
                    filled_data[t][nan_mask] = interpolated_values
            else:
                print(f"时间步 {t} 的有效数据点不足，使用临近插值...")

                if t > 0 and t < len(time) - 1:
                    prev_slice = filled_data[t - 1, :, :]
                    next_slice = missed_data[t + 1, :, :]

                    temp_slice = np.copy(time_slice)
                    for i in range(temp_slice.shape[0]):
                        for j in range(temp_slice.shape[1]):
                            if np.isnan(temp_slice[i, j]):
                                prev_val = prev_slice[i, j]
                                next_val = next_slice[i, j]

                                if not np.isnan(prev_val) and not np.isnan(next_val):
                                    temp_slice[i, j] = (prev_val + next_val) / 2
                                elif not np.isnan(prev_val):
                                    temp_slice[i, j] = prev_val
                                elif not np.isnan(next_val):
                                    temp_slice[i, j] = next_val


                    filled_data[t, :, :] = temp_slice

                    if np.isnan(filled_data[t]).any():
                        temp_slice = filled_data[t]
                        valid_mask = ~np.isnan(temp_slice)
                        if np.sum(valid_mask) >= 2:
                            valid_lats = lat_grid[valid_mask]
                            valid_lons = lon_grid[valid_mask]
                            valid_values = temp_slice[valid_mask]

                            nan_mask = np.isnan(temp_slice)
                            nan_lats = lat_grid[nan_mask]
                            nan_lons = lon_grid[nan_mask]

                            interpolated_values = griddata(
                                np.column_stack((valid_lons, valid_lats)),
                                valid_values,
                                np.column_stack((nan_lons, nan_lats)),
                                method='nearest'
                            )
                            filled_data[t][nan_mask] = interpolated_values
                elif t == 0:
                    next_slice = filled_data[t + 1, :, :]
                    temp_slice = np.copy(time_slice)
                    nan_mask = np.isnan(temp_slice)
                    temp_slice[nan_mask] = next_slice[nan_mask]
                    filled_data[t, :, :] = temp_slice
                else:
                    prev_slice = filled_data[t - 1, :, :]
                    temp_slice = np.copy(time_slice)
                    nan_mask = np.isnan(temp_slice)
                    temp_slice[nan_mask] = prev_slice[nan_mask]
                    filled_data[t, :, :] = temp_slice

    return filled_data


def visualize_interpolation(original, interpolated, time, lat, lon, time_indices=[0, 5, 9]):

    for t_idx in time_indices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axes[0].imshow(original[t_idx], cmap='viridis')
        axes[0].set_title(f'Original Data at time={time[t_idx]}')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(interpolated[t_idx], cmap='viridis')
        axes[1].set_title(f'Interpolated Data at time={time[t_idx]}')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

def process_interplation(origin):

    print(f"原始数据中的缺失值比例: {np.isnan(origin).sum() / origin.size:.2f}")

    filled = spatiotemporal_kriging_interpolation(origin, time, lat, lon)

    print(f"插值后数据中的缺失值比例: {np.isnan(filled).sum() / filled.size:.2f}")

    mask = ~np.isnan(origin)
    original_values = origin[mask]
    interpolated_values = filled[mask]
    mse = np.mean((original_values - interpolated_values) ** 2)
    rmse = np.sqrt(mse)
    print(f"非缺失值的RMSE: {rmse:.4f}")

    visualize_interpolation(origin, filled, time, lat, lon)

    return filled



if __name__ == "__main__":



    smap_filled = process_interplation(smap_sss)
    np.save('smap_filled.npy', smap_filled)
    anc_filled = process_interplation(anc_sst)
    np.save('anc_filled.npy', anc_filled)

    anc_filled = np.load('anc_filled.npy')
    # mean = np.mean(anc_filled)
    # std = np.std(anc_filled)
    # anc_normalized = (anc_filled - mean) / std
    # np.save('anc_normalized.npy', anc_normalized)

    smap_filled = np.load('smap_filled.npy')
    # mean = np.mean(smap_filled)
    # std = np.std(smap_filled)
    # smap_normalized = (smap_filled - mean) / std
    # np.save('smap_normalized.npy', smap_normalized)

    anc_filled = anc_filled.reshape(3600, 64)
    smap_filled = smap_filled.reshape(3600, 64)

    yangtze = np.stack((anc_filled, smap_filled), axis = 2)

    output_file = "yangtze_processed.nc"
    dataset = nc.Dataset(output_file, "w", format="NETCDF4")

    dataset.createDimension("time", 3600)
    dataset.createDimension("feature", 64)
    dataset.createDimension("channel", 2)

    data_var = dataset.createVariable("data", "f4", ("time", "feature", "channel"))

    data_var[:] = yangtze

    dataset.close()