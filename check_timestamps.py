import os
import pickle
import numpy as np
from datetime import datetime, timedelta

# Path to your data files
data_dir = "./shortpredict/val_data/eastsea_monthly"

# Load test indices
index_path = os.path.join(data_dir, "monthly_index_in_12_out_6.pkl")
with open(index_path, 'rb') as f:
    index_data = pickle.load(f)
    test_indices = index_data['test']
    print(f"Test samples: {len(test_indices)}")
    print(f"First test sample indices: {test_indices[0]}")

# Load the original data processing script variables to get start date
start_date = datetime(2015, 5, 4)

# Function to calculate monthly timestamps (simplified from data_process.py)
def calculate_timestamps(start_date, num_months):
    """Estimate monthly timestamps starting from the given date"""
    timestamps = []
    current_date = start_date
    for _ in range(num_months):
        # Get the last day of the current month
        _, last_day = calendar.monthrange(current_date.year, current_date.month)
        # Set date to the 15th of the month
        monthly_date = datetime(current_date.year, current_date.month, 15)
        timestamps.append(monthly_date)
        
        # Move to the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    return timestamps

# We need to estimate how many months are in the dataset
# This would normally come from the data file, but we'll use a reasonable estimate
# The monthly_data_in_12_out_6.pkl likely has this information
try:
    # Try to load the processed data to get the actual number of months
    data_path = os.path.join(data_dir, "monthly_data_in_12_out_6.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        processed_data = data['processed_data']
        print(f"Processed data shape: {processed_data.shape}")
        total_months = processed_data.shape[0]
except Exception as e:
    print(f"Error loading processed data: {e}")
    # Fallback: Estimate based on original dataset (3600 days ≈ 120 months)
    total_months = 120

print(f"Estimated total months: {total_months}")

# Generate timestamps
try:
    import calendar
    timestamps = calculate_timestamps(start_date, total_months)
    
    # Print sample dates for a few test indices
    for i, idx in enumerate(test_indices[:5]):
        history_start, history_end, future_end = idx
        
        history_start_date = timestamps[history_start] if history_start < len(timestamps) else "Unknown"
        history_end_date = timestamps[history_end-1] if history_end-1 < len(timestamps) else "Unknown"
        future_end_date = timestamps[future_end-1] if future_end-1 < len(timestamps) else "Unknown"
        
        print(f"Test sample {i}:")
        print(f"  History: {history_start_date} to {history_end_date}")
        print(f"  Prediction: {history_end_date} to {future_end_date}")
        print("  Predicted months:")
        
        # Show each predicted month
        for j in range(history_end, future_end):
            month_date = timestamps[j] if j < len(timestamps) else "Unknown"
            print(f"    Month {j-history_end+1}: {month_date}")
            
except Exception as e:
    print(f"Error generating timestamps: {e}") 