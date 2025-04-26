import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
train_data = pd.read_csv('/statistical-modeling/shortpredict/logs/2025-04-20-18-37-49-eastsea_monthly/train.csv')
test_data = pd.read_csv('/statistical-modeling/shortpredict/logs/2025-04-20-18-37-49-eastsea_monthly/test.csv')

# Set figure size
plt.figure(figsize=(12, 8))

# Plot training and validation loss
plt.subplot(2, 2, 1)
plt.plot(train_data['train_loss'], label='Training Loss')
plt.plot(train_data['valid_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot RMSE metrics
plt.subplot(2, 2, 2)
plt.plot(train_data['train_rmse'], label='Training RMSE')
plt.plot(train_data['valid_rmse'], label='Validation RMSE')
plt.title('RMSE Curves')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

# Plot MAPE metrics
plt.subplot(2, 2, 3)
plt.plot(train_data['train_mape'], label='Training MAPE')
plt.plot(train_data['valid_mape'], label='Validation MAPE')
plt.title('MAPE Curves')
plt.xlabel('Epoch')
plt.ylabel('MAPE')
plt.legend()
plt.grid(True)

# Plot WMAPE metrics
plt.subplot(2, 2, 4)
plt.plot(train_data['train_wmape'], label='Training WMAPE')
plt.plot(train_data['valid_wmape'], label='Validation WMAPE')
plt.title('WMAPE Curves')
plt.xlabel('Epoch')
plt.ylabel('WMAPE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)

# Plot test metrics by forecast horizon
plt.figure(figsize=(10, 6))
horizons = np.arange(1, len(test_data))

plt.plot(horizons, test_data.loc[0:len(horizons)-1, 'test_loss'], 'o-', label='MAE')
plt.plot(horizons, test_data.loc[0:len(horizons)-1, 'test_rmse'], 's-', label='RMSE')
plt.plot(horizons, test_data.loc[0:len(horizons)-1, 'test_mape']*100, '^-', label='MAPE (%)')
plt.plot(horizons, test_data.loc[0:len(horizons)-1, 'test_wmape']*100, 'd-', label='WMAPE (%)')

plt.title('Test Metrics by Forecast Horizon')
plt.xlabel('Forecast Horizon (months)')
plt.ylabel('Error Metrics')
plt.grid(True)
plt.legend()
plt.xticks(horizons)
plt.savefig('test_horizon_metrics.png', dpi=300)

print('Plots generated successfully') 