import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Given parameters
beta = 0.956
R_f = 1.04
delta = 0.07
alpha = 0.70
Z = 1
mu = 1  # Since 1 - mu = 0
l = 0.5
k_0 = 1


def simulate_firm_dynamics(k_0, Z, mu, alpha, beta, delta, R_f, l):
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

    k_path, d_path = [k_0], []
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

    return k_path[:-1], d_path  # Adjusting to make the lengths match


# Number of simulations
num_simulations = 1000

# Collecting results
all_k_paths, all_d_paths = [], []
for _ in range(num_simulations):
    k_path, d_path = simulate_firm_dynamics(k_0, Z, mu, alpha, beta, delta, R_f, l)
    all_k_paths.append(k_path)
    all_d_paths.append(d_path)

# Assuming each path's length is the same, we take the final values for simplicity
# For more complex analysis, consider aggregating all time points or specific ones
final_ks = [path[-1] for path in all_k_paths]
final_ds = [path[-1] for path in all_d_paths]

# Creating DataFrame
df_results = pd.DataFrame({"Final Capital": final_ks, "Final Dividends": final_ds})

# Computing statistics
mean_final_k = df_results["Final Capital"].mean()
var_final_k = df_results["Final Capital"].var()
mean_final_d = df_results["Final Dividends"].mean()
var_final_d = df_results["Final Dividends"].var()

# Saving results
stats_results = pd.DataFrame(
    {
        "Statistic": [
            "Mean Capital",
            "Variance Capital",
            "Mean Dividends",
            "Variance Dividends",
        ],
        "Value": [mean_final_k, var_final_k, mean_final_d, var_final_d],
    }
)
stats_results.to_csv("simulation_statistics.csv", index=False)

print("Statistics saved to 'simulation_statistics.csv'")
