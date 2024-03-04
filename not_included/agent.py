# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def model(
    time_periods,
    num_firms,
    short_term_investment_cost,
    long_term_investment_cost,
    short_term_tfp_increase,
    long_term_tfp_increase,
    long_term_investment_time,
    long_term_investment_prob,
):
    # Create variables to store results
    time = np.arange(time_periods)
    firm_tfp = np.zeros((num_firms, time_periods))
    aggregate_data = pd.DataFrame(
        columns=[
            "Time",
            "Mean_TFP",
            "Num_Long_Term_Investors",
            "Num_Short_Term_Investors",
        ]
    )

    firm_productivity = np.ones(num_firms)
    firm_research_investment = np.zeros(
        (num_firms, 3)
    )  # 0th column stores short term investment, 1st column stores long term investment, 2nd column stores investment time left for long term investment

    # distribution of returns
    # long term
    long_term_investment_mean_return = (
        long_term_tfp_increase  # Mean return on long-term investment
    )
    long_term_investment_std_return = (
        0.08  # Standard deviation of return on long-term investment
    )

    # short term
    short_term_investment_mean_return = (
        short_term_tfp_increase  # Mean return on long-term investment
    )
    short_term_investment_std_return = (
        0.05  # Standard deviation of return on long-term investment
    )

    # Loop over time periods
    for t in range(time_periods):
        # Compute TFP for each firm at time t
        firm_tfp[:, t] = firm_productivity * (
            1
            + firm_research_investment[:, 0] * short_term_tfp_increase
            + firm_research_investment[:, 1] * long_term_tfp_increase
        )

        # Compute investment choices for each firm at time t

        # Availabilty
        # long term investiment
        long_term_investment_return = np.random.binomial(
            n=1, p=long_term_investment_prob, size=num_firms
        )

        # Return
        # short term investiment return
        short_term_tfp_increase = np.random.normal(
            short_term_investment_mean_return,
            short_term_investment_std_return,
            num_firms,
        )

        # Return of the long term increse
        long_term_tfp_increase = np.random.normal(
            long_term_investment_mean_return, long_term_investment_std_return, num_firms
        )

        num_long_term_investors = 0
        num_short_term_investors = 0
        for i in range(num_firms):
            # Compute expected returns for short and long term investments
            if (
                firm_research_investment[i, 2] == 0
            ):  # if long-term investment finished, recalculate expected returns
                short_term_expected_return = (
                    firm_tfp[i, t] * short_term_tfp_increase[i]
                    - short_term_investment_cost
                )
                long_term_expected_return = (
                    firm_tfp[i, t] * long_term_tfp_increase[i]
                    - long_term_investment_cost
                )
            else:  # if long-term investment ongoing, expected return is zero
                short_term_expected_return = (
                    firm_tfp[i, t] * short_term_tfp_increase[i]
                    - short_term_investment_cost
                )
                long_term_expected_return = 0

            # Choose investment with highest expected return
            if long_term_expected_return > short_term_expected_return:
                firm_research_investment[i, 1] += 1
                firm_research_investment[i, 0] = 0
                firm_research_investment[i, 2] = long_term_investment_time
                num_long_term_investors += 1
            else:
                firm_research_investment[i, 0] += 1
                num_short_term_investors += 1

            # If long-term investment was made, adjust investment counter
            if (
                long_term_investment_return[i] == 1
                and firm_research_investment[i, 1] == 1
            ):
                firm_research_investment[i, 2] = long_term_investment_time - 1
            elif (
                firm_research_investment[i, 2] > 0
            ):  # reduce investment time left for long-term investment
                firm_research_investment[i, 2] -= 1

            # Compute new firm productivity for next period
            # new_firm_productivity = np.mean(firm_productivity)
            # firm_productivity = np.append(firm_productivity, new_firm_productivity)

            # Compute aggregate data at this time step
            mean_tfp = np.mean(firm_tfp[:, : t + 1], axis=1).mean()
            new_row = pd.DataFrame(
                {
                    "Time": [t],
                    "Firm": [i],
                    "TFP": [firm_tfp[i, t]],
                    "Mean_TFP": [mean_tfp],
                    "Num_Long_Term_Investors": [num_long_term_investors],
                    "Num_Short_Term_Investors": [num_short_term_investors],
                }
            )
            aggregate_data = pd.concat([aggregate_data, new_row], ignore_index=True)

    # Plot results
    plt.plot(time, np.mean(firm_tfp, axis=0))
    plt.xlabel("Time")
    plt.ylabel("Average TFP")
    plt.show()

    # Plot firm TFP over time
    for i in range(num_firms):
        plt.plot(time, firm_tfp[i])
    plt.title("Firm TFP over Time")
    plt.xlabel("Time")
    plt.ylabel("TFP")
    plt.show()
    return (aggregate_data, firm_tfp)


# Define parameters
time_periods = 50
num_firms = 3
short_term_investment_cost = 0.01
long_term_investment_cost = 0.04
short_term_tfp_increase = 0.02
long_term_tfp_increase = 0.2
long_term_investment_time = 3
long_term_investment_prob = 0.3

# Simulate model
df, firm_tfp = model(
    time_periods,
    num_firms,
    short_term_investment_cost,
    long_term_investment_cost,
    short_term_tfp_increase,
    long_term_tfp_increase,
    long_term_investment_time,
    long_term_investment_prob,
)


def marginal_revenue(demand_function, market_quantity, price):
    # Calculate the derivative of the demand function at the current market quantity
    h = 0.001
    derivative = (
        demand_function(market_quantity + h) - demand_function(market_quantity - h)
    ) / (2 * h)

    # Calculate the elasticity of demand at the current market quantity
    elasticity_of_demand = derivative * (market_quantity / price)

    # Calculate the marginal revenue for the firm
    marginal_revenue = price * (1 + (1 / elasticity_of_demand))

    return marginal_revenue


# Set the market demand function
def market_demand(price):
    return 5000 - 10 * price


def demand_function(quantity):
    return 500 - 0.1 * quantity


def calc_marginal_cost(tfp, y, w):
    a = tfp * y
    return w / a + 2 * (y / a)


def calc_cost(tfp, y):
    return 1 * y / tfp + y**2 / tfp


def simulate_market(firm_tfp, period):
    # Set the number of firms
    num_firms = firm_tfp.shape[0]

    # Set the initial price and quantities for each firm
    price = 10
    quantities = np.ones(num_firms) * market_demand(price) / num_firms
    market_power = np.zeros(num_firms)

    # Set the number of iterations for the simulation
    num_iterations = 50

    for t in range(num_iterations):
        # Calculate the total quantity produced by all firms
        total_quantity = np.sum(quantities)

        # Calculate the new market price based on the total quantity
        price = demand_function(total_quantity)

        # cancel the precedent data simulation
        new_rows = pd.DataFrame(
            columns=[
                "Time",
                "Firm",
                "TFP",
                "Marginal cost",
                "Marginal revenue",
                "Profit",
                "Quantity",
                "Price",
                "Market power",
            ]
        )

        # Update the quantities for each firm based on the new price

        for j in range(num_firms):
            # Calculate the marginal revenue for firm j
            marginal_revenue_j = marginal_revenue(
                demand_function, total_quantity, price
            )

            # Calculate the marginal cost for firm j
            marginal_cost = calc_marginal_cost(firm_tfp[j], quantities[j], 1)

            # Update the quantity for firm j based on the marginal revenue and marginal cost

            if marginal_revenue_j > marginal_cost:
                quantities[j] += 1
            elif marginal_revenue_j < marginal_cost:
                quantities[j] -= 1

            # Calculate the profit for firm j
            revenue = quantities[j] * price
            cost = calc_cost(firm_tfp[j], quantities[j])
            profit = revenue - cost

            # Calculate the market power for firm j
            market_power[j] = quantities[j] / total_quantity

            # Append results to dataframe
            new_row = pd.DataFrame(
                {
                    "Time": period,
                    "Firm": j,
                    "TFP": firm_tfp[j],
                    "Marginal cost": marginal_cost,
                    "Marginal revenue": marginal_revenue_j,
                    "Profit": profit,
                    "Quantity": quantities[j],
                    "Price": price,
                    "Market power": market_power[j],
                },
                index=[0],
            )
            new_rows = pd.concat([new_rows, new_row], ignore_index=True)

        # Append only the last row of new_rows to dataframe
        results = pd.DataFrame(new_rows.tail(num_firms))

    return results


market_power = np.zeros(num_firms)

results_list = []
for t in range(time_periods):
    firm_tfp_period = firm_tfp[:, t]
    period_results = simulate_market(firm_tfp_period, t)
    results_list.append(period_results)

results = pd.concat(results_list, ignore_index=True)
print(results)


# group data by firm
grouped = results.groupby("Firm")
grouped_mean = grouped["Quantity"].mean()
grouped_var = grouped["Quantity"].var()
q_norm = grouped_var.l(grouped["Quantity"] - grouped_mean) / math.sqrt(
    grouped_var / time_periods
)
normalized_data = grouped.apply(lambda x: (x["Quantity"] - x["Quantity"].mean()))
# plot time series of quantity and TFP for each firm
for firm, data in grouped:
    plt.plot(data["Time"], normalized_data, label=f"Firm {firm}")
    plt.plot(data["Time"], data["TFP"], label=f"Firm {firm} TFP")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Quantity/TFP")
    plt.title(f"Firm {firm} Quantity and TFP Time Series")
    plt.show()
