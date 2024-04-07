import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.4
beta = 0.965
Z = 1
l = 0.9
r_f = 1.05
delta = 0.1
# Assuming p is constant for this plot; adjust as necessary
p = (1.05 * l) / (1 - delta)


# Define the partial derivative as a function of u
def partial_derivative(u, mu=0.5):  # You can adjust mu as needed
    return (
        (1 / (1 - alpha))
        * ((Z * alpha * beta * (p * (1 - mu) + u)) / (p * (1 - alpha * beta * l)))
        ** (alpha / (1 - alpha))
        * (Z * alpha * beta * (1 - p))
        / (1 - alpha * beta * l)
    )


# Values of u from 0 to 1
u_values = np.linspace(0, 1, 100)

# Calculate partial derivative values
derivative_values = [partial_derivative(u) for u in u_values]
print(p)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(u_values, derivative_values, label="Partial Derivative w.r.t. μ")
plt.title("Partial Derivative of k-hat with respect to μ over u (0 to 1)")
plt.xlabel("u")
plt.ylabel("Partial Derivative Value")
plt.legend()
plt.grid(True)
plt.show()
