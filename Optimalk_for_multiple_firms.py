# Riccardo Dal Cero
# 16/01/23
# Simulate the cleansing effect model of the thesis
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


# Transtion function and policy function for capital and dividends
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


def reinvest_capital_with_min_return_exit(k_path_act, d_path_opt, Z, t, R_f):
    # Calculate return on capital for each firm at time t
    return_on_capital = np.divide(d_path_opt, k_path_act)

    # Identify the firm with the lowest return on capital
    min_return_index = np.argmin(return_on_capital)

    # Check if the lowest return is below the risk-free rate
    if return_on_capital[min_return_index] < R_f:
        exit_indices = [min_return_index]  # Only the firm with the lowest return exits
    else:
        exit_indices = []  # No firm exits if all are above the risk-free rate

    exit_rate = len(exit_indices) / len(k_path_act) if exit_indices else 0

    for exit_index in exit_indices:
        # Amount of capital to be reinvested from the exiting firm
        reinvestment_amount = k_path_act[exit_index]

        # Find the firm with the lowest capital to determine the new firm's capital
        lowest_capital_index = np.argmin(k_path_act)
        new_firm_capital = k_path_act[lowest_capital_index]

        # Calculate total return on capital for reinvestment proportions
        total_return_on_capital = (
            np.sum(return_on_capital) - return_on_capital[exit_index]
        )

        # Calculate reinvestment proportions based on return on capital
        reinvestment_proportions = np.divide(return_on_capital, total_return_on_capital)
        reinvestment_proportions[exit_index] = (
            0  # The exiting firm gets no reinvestment
        )

        # Distribute the reinvestment according to the calculated proportions
        for i in range(len(k_path_act)):
            if i != exit_index:
                k_path_act[i] += reinvestment_amount * reinvestment_proportions[i]

        # Replace the exiting firm's capital with the new firm's capital
        k_path_act[exit_index] = new_firm_capital

    return k_path_act, exit_rate


def main():
    STEP = 20
    NFIRM = 10
    K0 = np.ones(NFIRM)

    # Define the parameters for the productivity
    meanZ, std_devZ, lowerZ, upperZ = 0.05, 0.1, 0.01, 0.2
    aZ, bZ = (lowerZ - meanZ) / std_devZ, (upperZ - meanZ) / std_devZ
    Z = truncnorm.rvs(aZ, bZ, loc=meanZ, scale=std_devZ, size=NFIRM) + 1

    # Define the parameters for the leverage
    meanL, std_devL, lowerL, upperL = 0.5, 0.02, 0.01, 2
    aL, bL = (lowerL - meanL) / std_devL, (upperL - meanL) / std_devL
    L = truncnorm.rvs(aL, bL, loc=meanL, scale=std_devL, size=NFIRM)

    # Simulate the path to steady state
    k_path_opt = [np.array(K0)]
    k_path_act = [np.array(K0)]
    d_path_act = []
    K_path = [np.sum(Z * np.array(K0) ** alpha)]
    exit_rate = [0]
    for t in range(STEP):
        k_next_step = []
        d_current_step = []
        if (t % 2 == 0) and (t != 0):
            k_exit, exit_r = reinvest_capital_with_min_return_exit(
                k_path_act[-1], d_path_act[-1], Z, t, R_f
            )
            k_path_act[-1] = k_exit
        for i in range(NFIRM):
            k_current = k_path_act[-1][i]  # Use the last known capital for each firm
            k_next, d_current, k_hat = transition_and_policy(
                k_current, Z[i], mu, alpha, beta, delta, R_f, L[i]
            )
            k_next_step.append(k_next)
            d_current_step.append(d_current)
        k_path_opt.append(np.array(k_next_step))

        # Apply business cycle effects before moving to the next step
        k_next_step_adjusted, K = distribute_effect(Z, np.array(k_next_step), t)
        exit_r = 0

        k_path_act.append(k_next_step_adjusted)
        K_path.append(K)
        exit_rate.append(exit_r)
        d_path_act.append(np.array(d_current_step))

    # only for dividends
    k_next_step = []
    d_current_step = []

    for i in range(NFIRM):
        k_current = k_path_act[-1][i]  # Use the last known capital for each firm
        k_next, d_current, k_hat = transition_and_policy(
            k_current, Z[i], mu, alpha, beta, delta, R_f, L[i]
        )
        # k_next_step.append(k_next)
        d_current_step.append(d_current)
    d_path_act.append(np.array(d_current_step))
    # k_path_opt.append(np.array(k_next_step))
    return k_path_opt, d_path_act, k_path_act, Z, K_path, exit_rate


def BC(K, t):
    """Business Cycle effect on capital."""
    return K * (1 + 0.1 * math.sin(t))


def calculate_diffs(Z, k, t):
    """Calculate the diffs for all firms."""
    k = k**alpha
    K = np.sum(Z * k)  # Element-wise multiplication
    K_bc = BC(K, t)
    diff = K_bc - K
    return diff, K_bc


def distribute_effect(Z, k, t):
    """Distribute the total effect of the business cycle inversely by productivity."""
    total_diff, K = calculate_diffs(Z, k, t)

    # Calculate proportional effects based on productivity (or some rule)
    proportions = Z / np.sum(
        Z
    )  # For simplicity, directly proportional to productivity here

    # You might want to adjust this to make it inversely proportional or follow another distribution rule

    # Calculate how much each firm's capital should be adjusted
    adjustment = (
        total_diff * proportions
    )  # This will be an array of adjustments for each firm

    # Adjust capital for each firm
    new_k = k - adjustment

    return new_k, K


mu = 1  # Since 1 - mu = 0.25
beta = 0.956
R_f = 1.04
delta = 0.07
alpha = 0.70

k_path_opt, d_path_act, k_path_act, Z, K, exit_rate = main()
k_path_opt = np.array(k_path_opt)
k_path_act = np.array(k_path_act)
d_path_act = np.array(d_path_act)
K = np.array(K)
exit_rate = np.array(exit_rate)

# Flatten the arrays to create a long format DataFrame
num_steps = len(k_path_opt)
num_firms = k_path_opt.shape[1]
steps = np.repeat(np.arange(num_steps), num_firms)
firms = np.tile(np.arange(num_firms), num_steps)
k_tot = np.repeat(K, num_firms)
exit_rate = np.repeat(exit_rate, num_firms)
# Prepare the data dictionary
data = {
    "Step": steps,
    "Firm": firms,
    "Optimal K": k_path_opt.flatten(),
    "Actual K": k_path_act.flatten(),
    "Dividends": d_path_act.flatten(),
    "Total Production": k_tot,
    "Exit Rate": exit_rate,
    "Average productivity": k_tot / 10,
}

# Create the DataFrame
df = pd.DataFrame(data)


# Set up the plot
plt.figure(figsize=(14, 7))

# Plot Optimal and Actual K for each firm
for firm_id in df["Firm"].unique():
    firm_data = df[df["Firm"] == firm_id]

    plt.plot(
        firm_data["Step"],
        firm_data["Optimal K"],
        label=f"Optimal K Firm {firm_id}",
        linestyle="-",
        alpha=0.5,
    )
    plt.plot(
        firm_data["Step"],
        firm_data["Actual K"],
        label=f"Actual K Firm {firm_id}",
        linestyle="--",
        alpha=0.5,
    )

plt.xlabel("Step")
plt.ylabel("Capital")
plt.title("Optimal and Actual Capital Over Time for All Firms")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
# plt.savefig("in_out_market/literature/theory/main/OptimalK_noexit.png")
# Set up the plot for dividends
plt.figure(figsize=(14, 7))

# Plot Optimal and Actual K for each firm
for firm_id in df["Firm"].unique():
    firm_data = df[df["Firm"] == firm_id]

    plt.plot(
        firm_data["Step"],
        firm_data["Dividends"],
        label=f"Dividends {firm_id}",
        linestyle="-",
        alpha=0.5,
    )
    plt.plot(
        firm_data["Step"],
        firm_data["Actual K"],
        label=f"Actual K Firm {firm_id}",
        linestyle="--",
        alpha=0.5,
    )

plt.xlabel("Step")
plt.ylabel("d,k")
plt.title("Dividends and Actual Capital Over Time for All Firms")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
# plt.savefig("in_out_market/literature/theory/main/div_cap_noexit.png")
# plt.show()

# Set up the plot for dividends
plt.figure(figsize=(14, 7))


# Plot Optimal and Actual K for each firm
for firm_id in df["Firm"].unique():
    firm_data = df[df["Firm"] == firm_id]

    plt.plot(
        firm_data["Step"],
        firm_data["Total Production"],
        label=f"K",
        linestyle="-",
        alpha=0.5,
    )

    break


plt.xlabel("Step")
plt.ylabel("Output")
plt.title("Overall output Over Time for All Firms")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
# plt.show()
# plt.savefig("in_out_market/literature/theory/main/div_cap_noexit.png")
for firm_id in df["Firm"].unique():
    firm_data = df[df["Firm"] == firm_id]
    plt.plot(
        firm_data["Step"],
        firm_data["Exit Rate"],
        label=f"K",
        linestyle="-",
        alpha=0.5,
    )
    break
plt.xlabel("Step")
plt.ylabel("Exit rate")
plt.title("Exit rate over time")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
# plt.show()
# plt.savefig('in_out_market/literature/theory/main/div_cap_noexit.png')
plt.clf()
for firm_id in df["Firm"].unique():
    firm_data = df[df["Firm"] == firm_id]
    plt.plot(
        firm_data["Step"],
        firm_data["Average productivity"],
        label=f"k",
        linestyle="-",
        alpha=0.5,
    )
    break
plt.xlabel("Step")
plt.ylabel("K/10")
plt.title("Average productivity over time")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
# plt.savefig('in_out_market/literature/theory/main/Avg_producitity_noexit.png')
