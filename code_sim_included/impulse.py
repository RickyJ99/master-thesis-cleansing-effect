import numpy as np
import pandas as pd

# Parameters
delta = 0.07
alpha = 0.70
Z = 1
beta = 0.95
R_f = 0.04
k_0 = -0.01
lambda_initial = 0.1

# Generate p and l
np.random.seed(0)  # For reproducibility
p = 0.99  # np.clip(np.random.normal(0.5, 0.1), 0, 1)
l = 0.01  # np.clip(np.random.normal(0.5, 0.1), 0, 1)

# Calculate a and lambda
lambda_value = 1 / (1 - l)
a = (
    p - p * l - beta * p + beta * delta * p + beta * R_f * l + p - delta * p - R_f * l
) / (beta * p)

# Number of periods for simulation
periods = 50
# benchmark
# Initialize arrays to store paths
k_path = np.zeros(periods)
d_path = np.zeros(periods)
k_path[0] = k_0

# Calculate initial d_0*
d_path[0] = (a * lambda_value * k_0) / (1 + beta * lambda_value)

# Simulate the paths
for t in range(1, periods):
    k_path[t] = (
        k_path[t - 1]
        * (a * lambda_value * (1 + (beta - 1) * lambda_value))
        / (1 + beta * lambda_value)
    )
    d_path[t] = (a * lambda_value * k_path[t]) / (1 + beta * lambda_value)

# COMPARISON
# Increase productivity Z by 10% and recalculate paths
Z_new = Z * 0.8
k_0 = -0.01
p = 0.1
l = 1.5
a_new = (
    p - p * l - beta * p + beta * delta * p + beta * R_f * l + p - delta * p - R_f * l
) / (beta * p)
k_path_new = np.zeros(periods)
d_path_new = np.zeros(periods)
k_path_new[0] = k_0
d_path_new[0] = (a_new * lambda_value * k_0) / (1 + beta * lambda_value)

for t in range(1, periods):
    k_path_new[t] = (
        k_path_new[t - 1]
        * (a_new * lambda_value * (1 + (beta - 1) * lambda_value))
        / (1 + beta * lambda_value)
    )
    d_path_new[t] = (a_new * lambda_value * k_path_new[t]) / (1 + beta * lambda_value)

# Create a dataframe
df = pd.DataFrame(
    {
        "Period": np.arange(periods),
        "Capital": k_path,
        "Capital10": k_path_new,
        "Consumption": d_path,
        "Consumption10": d_path_new,
    }
)

# Export the dataframe to a CSV file
csv_path = "output_data/impulse_response.csv"
df.to_csv(csv_path, index=False)
