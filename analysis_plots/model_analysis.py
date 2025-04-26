import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns

# Load data
train_data = pd.read_csv('/statistical-modeling/shortpredict/logs/2025-04-20-18-37-49-eastsea_monthly/train.csv')
test_data = pd.read_csv('/statistical-modeling/shortpredict/logs/2025-04-20-18-37-49-eastsea_monthly/test.csv')

# 1. Convergence Analysis
plt.figure(figsize=(10, 6))
# Apply Savitzky-Golay filter to smooth the curves
train_loss_smooth = savgol_filter(train_data['train_loss'], 21, 3)
valid_loss_smooth = savgol_filter(train_data['valid_loss'], 21, 3)

plt.plot(train_data.index, train_data['train_loss'], 'b-', alpha=0.3, label='Training Loss (Raw)')
plt.plot(train_data.index, train_data['valid_loss'], 'r-', alpha=0.3, label='Validation Loss (Raw)')
plt.plot(train_data.index, train_loss_smooth, 'b-', linewidth=2, label='Training Loss (Smoothed)')
plt.plot(train_data.index, valid_loss_smooth, 'r-', linewidth=2, label='Validation Loss (Smoothed)')

# Find best validation epoch
best_epoch = train_data['valid_loss'].idxmin()
best_val_loss = train_data['valid_loss'].min()
plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch: {best_epoch}')
plt.text(best_epoch+5, best_val_loss, f'Best Val Loss: {best_val_loss:.4f}', fontsize=12)

plt.title('Convergence Analysis: Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('convergence_analysis.png', dpi=300)

# 2. Learning Dynamics Analysis
plt.figure(figsize=(12, 10))

# Define a list of metrics and their corresponding colors
metrics = ['loss', 'rmse', 'mape', 'wmape']
colors = ['blue', 'green', 'red', 'purple']

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    
    # Calculate rate of improvement (first derivative)
    train_metric = train_data[f'train_{metric}'].values
    valid_metric = train_data[f'valid_{metric}'].values
    
    # Use diff to get the improvement rate
    train_improvement = np.diff(train_metric)
    valid_improvement = np.diff(valid_metric)
    
    # Apply smoothing for better visualization
    window = 15 if len(train_improvement) > 30 else 7
    train_smooth = savgol_filter(train_improvement, window, 3)
    valid_smooth = savgol_filter(valid_improvement, window, 3)
    
    plt.plot(train_data.index[1:], train_improvement, color=colors[i], alpha=0.3, label=f'Train {metric.upper()} Change (Raw)')
    plt.plot(train_data.index[1:], valid_improvement, color='orange', alpha=0.3, label=f'Valid {metric.upper()} Change (Raw)')
    plt.plot(train_data.index[1:], train_smooth, color=colors[i], label=f'Train {metric.upper()} Change (Smoothed)')
    plt.plot(train_data.index[1:], valid_smooth, color='orange', label=f'Valid {metric.upper()} Change (Smoothed)')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'{metric.upper()} Improvement Rate')
    plt.xlabel('Epoch')
    plt.ylabel(f'Change in {metric.upper()} per Epoch')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('learning_dynamics.png', dpi=300)

# 3. Error Distribution Across Horizons
plt.figure(figsize=(12, 8))

# Extract metrics by forecast horizon
horizons = range(1, len(test_data))
metrics_by_horizon = {
    'MAE': test_data.loc[0:len(horizons)-1, 'test_loss'].values,
    'RMSE': test_data.loc[0:len(horizons)-1, 'test_rmse'].values,
    'MAPE': test_data.loc[0:len(horizons)-1, 'test_mape'].values * 100,  # Convert to percentage
    'WMAPE': test_data.loc[0:len(horizons)-1, 'test_wmape'].values * 100  # Convert to percentage
}

# Normalize metrics for comparison
normalized_metrics = {}
for metric_name, values in metrics_by_horizon.items():
    normalized_metrics[metric_name] = (values - np.min(values)) / (np.max(values) - np.min(values))

# Create a DataFrame for better plotting
horizon_df = pd.DataFrame({
    'Horizon': list(horizons) * 4,
    'Metric': sum([[metric] * len(horizons) for metric in normalized_metrics.keys()], []),
    'Normalized Value': np.concatenate(list(normalized_metrics.values()))
})

# Line plot for normalized metrics
plt.subplot(1, 2, 1)
for metric, color in zip(normalized_metrics.keys(), ['blue', 'orange', 'green', 'red']):
    plt.plot(horizons, normalized_metrics[metric], 'o-', color=color, label=f'Normalized {metric}')
    
    # Linear regression to show trend
    z = np.polyfit(horizons, normalized_metrics[metric], 1)
    p = np.poly1d(z)
    plt.plot(horizons, p(horizons), '--', color=color, alpha=0.5)

plt.title('Normalized Error Metrics by Forecast Horizon')
plt.xlabel('Forecast Horizon (months)')
plt.ylabel('Normalized Error (0-1)')
plt.grid(True)
plt.legend()

# Heatmap of original metrics
plt.subplot(1, 2, 2)
error_matrix = np.array([metrics_by_horizon['MAE'], 
                         metrics_by_horizon['RMSE'],
                         metrics_by_horizon['MAPE'],
                         metrics_by_horizon['WMAPE']])

ax = sns.heatmap(error_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=horizons,
            yticklabels=['MAE', 'RMSE', 'MAPE (%)', 'WMAPE (%)'])
plt.title('Error Metrics Heatmap by Horizon')
plt.xlabel('Forecast Horizon (months)')

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300)

# 4. Training Efficiency Analysis
plt.figure(figsize=(10, 6))

# Calculate rolling mean and standard deviation
window = 10
metrics = ['loss', 'rmse']
colors = ['blue', 'green']

for i, metric in enumerate(metrics):
    train_rolling_mean = train_data[f'train_{metric}'].rolling(window=window).mean()
    train_rolling_std = train_data[f'train_{metric}'].rolling(window=window).std()
    
    valid_rolling_mean = train_data[f'valid_{metric}'].rolling(window=window).mean()
    valid_rolling_std = train_data[f'valid_{metric}'].rolling(window=window).std()
    
    # Skip the first window-1 NaN values from rolling calculation
    start_idx = window - 1
    
    plt.subplot(1, 2, i+1)
    
    # Plot mean with shaded standard deviation
    epochs = train_data.index[start_idx:]
    
    plt.plot(epochs, train_rolling_mean[start_idx:], color=colors[i], label=f'Train {metric.upper()} (Mean)')
    plt.fill_between(epochs, 
                     train_rolling_mean[start_idx:] - train_rolling_std[start_idx:],
                     train_rolling_mean[start_idx:] + train_rolling_std[start_idx:],
                     color=colors[i], alpha=0.2)
    
    plt.plot(epochs, valid_rolling_mean[start_idx:], color='red', label=f'Valid {metric.upper()} (Mean)')
    plt.fill_between(epochs, 
                     valid_rolling_mean[start_idx:] - valid_rolling_std[start_idx:],
                     valid_rolling_mean[start_idx:] + valid_rolling_std[start_idx:],
                     color='red', alpha=0.2)
    
    plt.title(f'{metric.upper()} with Uncertainty')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric.upper()}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('training_efficiency.png', dpi=300)

print("Advanced analysis plots generated successfully!") 