# Simulation of optimal path in case of frictions
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Given values
l = 0.5  # Leverage
R_f = 1.04  # Risk-free rate
delta = 0.07  # Depreciation rate

# Calculate p using the given formula
p = (l * R_f) / (1 - delta)
p


# Given parameters
beta = 0.956
R_f = 1.04
delta = 0.07
alpha = 0.70
Z = 1
mu = 1  # Since 1 - mu = 0.25
l = 0.5
k_0 = 1


# Transition function and policy function for capital and dividends
def transition_and_policy(k_0, Z, mu, alpha, beta, delta, R_f, l):
    k_1 = (
        (Z * mu * k_0**alpha * (1 - delta))
        / (l * R_f)
        * (alpha * beta)
        / (1 - l * alpha * beta)
    )
    d_0 = (
        (Z * mu * k_0**alpha * (1 - delta))
        / (l * R_f)
        * (1 - alpha * beta)
        / (1 - l * alpha * beta)
        * beta
    )
    k_hat = (
        (Z * mu * (1 - delta)) / (l * R_f) * (alpha * beta) / (1 - l * alpha * beta)
    ) ** (1 / (1 - alpha))
    return k_1, d_0, k_hat


# Simulate the path to steady state
k_path = [k_0]
d_path = []
k_current = k_0
while True:
    k_next, d_current, k_hat = transition_and_policy(
        k_current, Z, mu, alpha, beta, delta, R_f, l
    )
    k_path.append(k_next)
    d_path.append(d_current)
    if np.abs(k_next - k_current) < 1e-6:  # Convergence criterion
        break
    k_current = k_next

# Plotting
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(k_path, "-o", label="Capital Path")
plt.axhline(y=k_hat, color="r", linestyle="--", label="Steady State Capital")
plt.title("Optimal Path of Capital")
plt.xlabel("Time")
plt.ylabel("Capital")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(d_path, "-o", label="Dividend Path")
plt.title("Path of Dividends")
plt.xlabel("Time")
plt.ylabel("Dividends")
plt.legend()

plt.tight_layout()
plt.show()

# Adjusting the simulation data list lengths
k_path_adjusted = k_path[:-1]  # Remove the last k value to match the lengths

# Recreate the DataFrame with the adjusted data
simulation_data_adjusted = pd.DataFrame(
    {
        "Time": range(len(k_path_adjusted)),
        "Capital": k_path_adjusted,
        "Dividends": d_path,
    }
)

# Save the adjusted data to a CSV file for LaTeX/TikZ plotting
csv_file_path_adjusted = "simulation.csv"
simulation_data_adjusted.to_csv(csv_file_path_adjusted, index=False)

csv_file_path_adjusted
