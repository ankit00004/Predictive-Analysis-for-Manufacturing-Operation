# Randomly Generated dataset
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate synthetic dataset


def generate_synthetic_data(n_samples=500):
    data = pd.DataFrame({
        "Machine_ID": np.arange(1, n_samples + 1),
        # Temperature between 50 and 100
        "Temperature": np.random.uniform(50, 100, n_samples),
        # Run Time between 1 and 200 hours
        "Run_Time": np.random.uniform(1, 200, n_samples),
    })

    # Downtime_Flag based on conditions
    conditions = [
        # High temp and long runtime
        (data["Temperature"] > 85) & (data["Run_Time"] > 150),
        # Low temp but extreme runtime
        (data["Temperature"] < 60) & (data["Run_Time"] > 180),
    ]
    choices = [1, 1]  # 1 indicates downtime
    data["Downtime_Flag"] = np.select(
        conditions, choices, default=0)  # Default is no downtime

    return data


# Generate 5 datasets and save them
file_paths = []
for i in range(5):
    dataset = generate_synthetic_data()
    file_path = f"synthetic_manufacturing_data_{i+1}.csv"
    dataset.to_csv(file_path, index=False)
    file_paths.append(file_path)

# Print file paths to check the saved files
print(file_paths)
