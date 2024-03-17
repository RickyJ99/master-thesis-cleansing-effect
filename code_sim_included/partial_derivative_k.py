import pandas as pd
import numpy as np


# Define the function for the partial derivative with respect to l, correctly handling edge cases
def compute_partial_derivative_k_with_respect_to_l(
    alpha, beta, Z, mu, u, l_values, delta, R_f
):
    derivatives = []
    for l in l_values:
        # Avoid division by zero by setting derivative to NaN in edge cases
        if l == 0 or l == 1 / (alpha * beta):
            derivatives.append(np.nan)
            continue
        p = (R_f * l) / (1 - delta)
        factor1 = (Z * alpha * beta * (p * (1 - mu) + u)) / (p * (1 - alpha * beta * l))
        factor1_powered = factor1 ** (alpha / (1 - alpha))

        term1 = -(1 - delta) * mu - alpha * beta * Z * R_f * (1 - mu) * l / (
            R_f * l**2 * (1 - alpha * beta * l)
        )
        term2 = (
            alpha * beta * (1 - delta) * mu
            + alpha**2 * beta**2 * R_f * (1 - mu) * Z * l
        ) / (R_f * l * (1 - alpha * beta * l) ** 2)
        term3 = (alpha * beta * Z * (1 - mu)) / (l * (1 - alpha * beta * l))

        derivative = (1 / (1 - alpha)) * factor1_powered * (term1 + term2 + term3)
        derivatives.append(derivative)

    return derivatives


# Assuming the same parameters as before, we calculate the derivatives for a range of l values
l_values = np.linspace(0.01, 0.99, 100)
alpha = 0.1
beta = 0.956
R_f = 1.04
delta = 0.07
mu = 0.75
u = mu
Z = 1

adjusted_derivatives = compute_partial_derivative_k_with_respect_to_l(
    alpha, beta, Z, mu, u, l_values, delta, R_f
)

# Create a DataFrame for the adjusted derivatives
df_adjusted = pd.DataFrame({"l": l_values, "par": adjusted_derivatives})

# Export the DataFrame to a CSV file
adjusted_file_path = "output_data/adjusted_partial_derivative_k_with_respect_to_l_1.csv"
df_adjusted.to_csv(adjusted_file_path, index=False)

# Return the file path for download
adjusted_file_path
