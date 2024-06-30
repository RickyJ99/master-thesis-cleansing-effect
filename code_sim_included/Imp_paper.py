# Plotting the evolution of capital using quadratic preferences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initial log deviation of capital (10% deviation from the steady state)
# Parameters
DELTA = 0.07
ALPHA = 0.70
Z = 1
BETA = 0.9
PERIODS = 100

# variables
k_0 = np.log(1.1)
# checking if beta is = to the steady state prescription of capital

k_hat = ((Z * ALPHA * BETA) / (1 - BETA * (1 - DELTA))) ** (
    1 / (1 - ALPHA)
)  # steady state level of capital
coef_k = ((ALPHA * Z * k_hat ** (ALPHA - 1)) + 1 - DELTA) ** (
    -1
)  # should be equal to beta
d_hat = Z * k_hat**ALPHA - DELTA * k_hat  # steady state level of dividends

# Evolution of capital for 100 PERIODS in log deviations from the steady state
k_values = [k_0]
d_values = [(1 - BETA**2) / BETA * k_0]
for t in range(1, PERIODS):
    k_t = BETA * k_values[t - 1]
    k_values.append(k_t)
    d_t = (1 - BETA**2) / BETA * k_t
    d_values.append(d_t)


# Plotting the evolution of capital and dividends in log deviations from the steady state
plt.figure(figsize=(10, 6))
plt.plot(
    range(PERIODS),
    k_values,
    marker="o",
    linestyle="-",
    color="b",
    label="Capital (log deviations)",
)
plt.plot(
    range(PERIODS),
    d_values,
    marker="o",
    linestyle="-",
    color="g",
    label="Dividends (log deviations)",
)
plt.title(
    "Evolution of Capital and Dividends Over Time (Log Deviations from Steady State)"
)
plt.xlabel("Periods")
plt.ylabel("Log Deviations from Steady State")
plt.legend()
plt.grid(True)
# plt.savefig('capital_dividends_evolution.png')

# Create a DataFrame to hold the data
data = {
    "Period": range(PERIODS),
    "Capital (log deviations)": k_values,
    "Dividends (log deviations)": d_values,
    "Steady State Capital": [k_hat] * PERIODS,
    "Steady State Dividends": [d_hat] * PERIODS,
}
df = pd.DataFrame(data)
csv_path = "output_data/quadratic_function_lowerb.csv"
df.to_csv(csv_path, index=False)
