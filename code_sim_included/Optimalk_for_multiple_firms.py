# Riccardo Dal Cero
# 16/01/23
# Simulate the cleansing effect model of the thesis
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import truncnorm
from scipy import stats
import matplotlib.pyplot as plt


def calculate_p(R_f, l, delta):
    # compute p
    return (R_f * l) / (1 - delta)


def f(Z, k_0, alpha):
    """
    Placeholder for the production function f, dependent on capital k_0.
    This function needs to be defined based on your economic model.
    """
    return Z * (k_0**alpha)


# Transtion function and policy function for capital and dividends
def transition_and_policy(k_0, Z, mu, alpha, beta, delta, R_f, l):
    p = calculate_p(R_f, l, delta)
    # Calculate the common term in both equations for simplification
    common_term = ((p + mu - mu * p) / p) * Z * k_0**alpha

    # Calculate next period's capital (k_1) based on the given equation
    k_1 = common_term * (alpha * beta / (1 - l * alpha * beta))

    # Calculate optimal current period's dividends (d*_0) based on the given equation
    d_0 = common_term * ((1 - alpha * beta) / (1 - l * alpha * beta))
    return k_1, d_0


def transition_equation(Z, k_0, alpha, mu, delta, R_f, l, d_0_star, t):
    """
    Computes the next period's capital (k_1) based on the given transition equation.

    Parameters:
    - k_0: Current period's capital.
    - p: Price level.
    - mu: Marginal utility of consumption.
    - delta: Depreciation rate.
    - R_f: Risk-free rate.
    - l: Leverage.
    - d_0_star: Optimal current period's dividends.

    Returns:
    - k_1: Next period's capital.
    """
    p = calculate_p(R_f, l, delta)
    Z_b = BC(Z, t)
    # Calculate the terms inside the brackets
    term_1 = ((p + mu - mu * p) / p) * f(Z_b, k_0, alpha)
    term_2 = ((p - delta * p - R_f * l) / p) * k_0
    inside_brackets = term_1 + term_2 - d_0_star

    # Calculate next period's capital (k_1)
    k_1 = inside_brackets * ((1 - l) ** (-1))

    return k_1


def reinvest_capital_with_min_return_exit(
    k_path_act, d_path_opt, k_path_opt, Z, t, R_f
):
    # Calculate return on capital for each firm at time t
    return_on_capital = np.divide(d_path_opt, k_path_act) + np.divide(
        k_path_act - k_path_opt, k_path_opt
    )

    sorted_indices = np.argsort(return_on_capital)
    exit_indices = [i for i in sorted_indices if return_on_capital[i] < R_f][:2]

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


def main(exit):
    STEP = 20
    NFIRM = 10
    K0 = np.ones(NFIRM) * 2

    # Define the parameters for the productivity
    meanZ, std_devZ, lowerZ, upperZ = 0.05, 0.15, 0.01, 0.2
    aZ, bZ = (lowerZ - meanZ) / std_devZ, (upperZ - meanZ) / std_devZ
    Z = truncnorm.rvs(aZ, bZ, loc=meanZ, scale=std_devZ, size=NFIRM) + 1

    # Define the parameters for the leverage
    meanL, std_devL, lowerL, upperL = 0.5, 0.1, 0.01, 1
    aL, bL = (lowerL - meanL) / std_devL, (upperL - meanL) / std_devL
    L = truncnorm.rvs(aL, bL, loc=meanL, scale=std_devL, size=NFIRM)

    # Simulate the path to steady state
    k_path_opt = [np.array(K0)]
    k_path_act = [np.array(K0)]
    d_path_act = []
    K_path_act = [np.array(Z * np.array(K0) ** alpha)]
    K_path_opt = []  # [np.array(Z * np.array(K0) ** alpha)]
    exit_rate = [0]
    for t in range(STEP):
        k_next_step = []
        k_next_step_act = []
        K_next_step_opt = []
        K_next_step_act = []
        d_current_step = []
        exit_r = 0

        if (exit) and (t != 0):
            k_exit, exit_r = reinvest_capital_with_min_return_exit(
                k_path_act[-1], d_path_act[-1], k_path_act[-2], Z, t, R_f
            )
            k_path_act[-1] = k_exit
        for i in range(NFIRM):
            k_current = k_path_act[-1][i]  # Use the last known capital for each firm
            k_next, d_current = transition_and_policy(
                k_current, Z[i], mu, alpha, beta, delta, R_f, L[i]
            )
            k_next_step.append(k_next)
            d_current_step.append(d_current)
            # Apply business cycle effects before moving to the next step
            K_opt, K_act = calculate_diffs(Z[i], k_current, t)
            k_act = transition_equation(
                Z[i], k_current, alpha, mu, delta, R_f, L[i], d_current, t
            )
            k_next_step_act.append(k_act)
            K_next_step_act.append(K_act)
            K_next_step_opt.append(K_opt)

        k_path_opt.append(np.array(k_next_step))
        k_path_act.append(np.array(k_next_step_act))
        K_path_opt.append(np.array(K_next_step_opt))
        K_path_act.append(np.array(K_next_step_act))
        exit_rate.append(exit_r)
        d_path_act.append(np.array(d_current_step))

    # only for dividends
    k_next_step = []
    d_current_step = []

    for i in range(NFIRM):
        k_current = k_path_act[-1][i]  # Use the last known capital for each firm
        k_next, d_current = transition_and_policy(
            k_current, Z[i], mu, alpha, beta, delta, R_f, L[i]
        )
        # k_next_step.append(k_next)
        d_current_step.append(d_current)
    d_path_act.append(np.array(d_current_step))
    K_path_opt.append(K_path_opt[-1])

    return K_path_opt, d_path_act, K_path_act, Z, k_path_opt, k_path_act, exit_rate


def BC(Z, t):
    """Business Cycle effect on capital."""
    return Z * (1 + 0.1 * math.sin(t))


def calculate_diffs(Z, k, t):
    """Calculate the diffs for all firms."""
    ka = k**alpha
    K_opt = Z * ka  # Element-wise multiplication
    Z_bc = BC(Z, t)
    K_bc = Z_bc * ka
    return K_opt, K_bc


def compute_stats_by_step_and_firm_with_ci(df, confidence_level=0.95):
    """
    Compute the mean, variance, and confidence intervals at each step for each firm for specified columns.

    Parameters:
    - df: pandas DataFrame containing simulation data.
    - confidence_level: The confidence level for the confidence interval calculation.

    Returns:
    - A pandas DataFrame with the mean, variance, and confidence intervals of specified columns for each step and firm.
    """
    # Define columns to calculate mean and variance for
    columns = [
        "Optimal_k",
        "Actual_k",
        "Dividends",
        "Total_Production_actual",
        "Total_Production_optimal",
        "Exit_Rate",
    ]

    # Group by 'Step' and 'Firm', then calculate mean and variance
    grouped = df.groupby(["Step", "Firm"])[columns].agg(["mean", "var"])

    # Reset index to make 'Step' and 'Firm' columns again if they were part of the MultiIndex
    grouped = grouped.reset_index()

    # Now adjust the column names formatting
    grouped.columns = [
        "_".join(col).strip().replace(" ", "_") if col[1] else col[0]
        for col in grouped.columns.values
    ]

    # Merge with sample sizes.
    sample_sizes = df.groupby(["Step", "Firm"]).size().reset_index(name="N")
    grouped = pd.merge(grouped, sample_sizes, on=["Step", "Firm"])

    z_score = stats.norm.ppf((1 + confidence_level) / 2.0)

    for col in columns:
        # Prepare the column names for SEM and CI calculations
        mean_col = f"{col}_mean"
        var_col = f"{col}_var"
        sem_col = f"{col}_SEM"
        ci_lower_col = f"{col}_CI_Lower"
        ci_upper_col = f"{col}_CI_Upper"

        # Calculate SEM, CI Lower, and CI Upper
        grouped[sem_col] = np.sqrt(grouped[var_col]) / np.sqrt(grouped["N"])
        grouped[ci_lower_col] = grouped[mean_col] - z_score * grouped[sem_col]
        grouped[ci_upper_col] = grouped[mean_col] + z_score * grouped[sem_col]

    return grouped


mu = float(input("1-mu:"))  # 0.75  # Since 1 - mu = 0.25
beta = 0.956
R_f = 1.04
delta = 0.07
alpha = 0.70
exit = bool(int(input("Exit (0 or 1): ")))
num_runs = 100  # Number of simulations to run

# Lists to accumulate data
steps_list = []
firms_list = []
runs_list = []
k_path_opt_list = []
k_path_act_list = []
d_path_act_list = []
K_path_opt_list = []
K_path_act_list = []
exit_rate_list = []

# Initialize lists to store the simulation data
runs_list, steps_list, firms_list = [], [], []
k_path_opt_list, k_path_act_list, d_path_act_list = [], [], []
k_tot_list, exit_rate_list, total_production_optimal_list = [], [], []

for run_id in range(num_runs):
    K_path_opt, d_path_act, K_path_act, Z, k_path_opt, k_path_act, exit_rate = main(
        exit
    )  # K_path_opt, d_path_act, K_path_act, Z, k_path_opt, k_path_act, exit_rate
    k_path_opt = np.array(k_path_opt)
    k_path_act = np.array(k_path_act)
    K_path_opt = np.array(K_path_opt)
    K_path_act = np.array(K_path_act)
    d_path_act = np.array(d_path_act)
    exit_rate = np.array(exit_rate)

    # Flatten and repeat as necessary for long format
    num_steps = len(k_path_opt)
    num_firms = k_path_opt.shape[1]
    steps = np.repeat(np.arange(num_steps), num_firms)
    firms = np.tile(np.arange(num_firms), num_steps)
    exit_rate = np.repeat(exit_rate, num_firms)

    # Calculate Total_Production_optimal at each step (sum of Optimal_K for all firms)
    total_production_optimal = np.sum(K_path_opt, axis=1)
    # Repeat the Total_Production_optimal for each firm
    total_production_optimal_repeated = np.repeat(total_production_optimal, num_firms)

    # Calculate Total_Production_optimal at each step (sum of Optimal_K for all firms)
    total_production_actual = np.sum(K_path_act, axis=1)
    # Repeat the Total_Production_optimal for each firm
    total_production_actual_repeated = np.repeat(total_production_actual, num_firms)

    # Append data for this run
    runs_list.extend([run_id] * num_steps * num_firms)
    steps_list.extend(steps)
    firms_list.extend(firms)
    k_path_opt_list.extend(k_path_opt.flatten())
    k_path_act_list.extend(k_path_act.flatten())
    d_path_act_list.extend(d_path_act.flatten())
    K_path_opt_list.extend(total_production_optimal_repeated)
    K_path_act_list.extend(total_production_actual_repeated)
    exit_rate_list.extend(exit_rate)

# Prepare the data dictionary including Total_Production_optimal
data = {
    "Run": runs_list,
    "Step": steps_list,
    "Firm": firms_list,
    "Optimal_k": k_path_opt_list,
    "Actual_k": k_path_act_list,
    "Dividends": d_path_act_list,
    "Total_Production_actual": K_path_act_list,
    "Total_Production_optimal": K_path_opt_list,
    "Exit_Rate": exit_rate_list,
}

# Create the DataFrame
df = pd.DataFrame(data)
exit_str = "exit_" if exit else "noexit_"
friction_str = "nofriction" if mu == 1 else "friction"

df.to_csv(f"output_data/{exit_str}{friction_str}.csv", index=False)


print(f"Data from all runs saved to {exit_str}{friction_str}.csv")
stats_df = compute_stats_by_step_and_firm_with_ci(df)
# Specify your path correctly, especially if you're using directories
stats_df.to_csv(f"output_data/{exit_str}{friction_str}_stats.csv", index=False)
