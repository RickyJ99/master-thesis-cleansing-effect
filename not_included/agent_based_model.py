# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#cleaning effect of recession
#a model of growth of creative distrucion
#the cleanings effect of recession Caballero
def model(
    TIME_PERIODS,
    NUM_FIRMS,
    SHORT_TERM_INVESTIMENT_COST,
    LONG_TERM_INVESTIMENT_COST,
    SHORT_TERM_TFP_INCREASE,
    LONG_TERM_TFP_INCREASE,
    LONG_TERM_INVESTIMENT_TIME,
    LONG_TERM_INVESTIMENT_PROB,
):
    # Create variables to store results
    time = np.arange(TIME_PERIODS)
    firm_tfp = np.zeros((NUM_FIRMS, TIME_PERIODS))
    aggregate_data = pd.DataFrame(
        columns=[
            "Time",
            "Mean_TFP",
            "Num_Long_Term_Investors",
            "Num_Short_Term_Investors",
        ]
    )

    firm_productivity = np.ones(NUM_FIRMS)
    firm_research_investment = np.zeros(
        (NUM_FIRMS, 3)
    )  # 0th column stores short term investment, 1st column stores long term investment, 2nd column stores investment time left for long term investment

    # distribution of returns
    # long term
    long_term_investment_mean_return = (
        LONG_TERM_TFP_INCREASE  # Mean return on long-term investment
    )
    long_term_investment_std_return = (
        0.1  # Standard deviation of return on long-term investment
    )

    # short term
    short_term_investment_mean_return = (
        SHORT_TERM_TFP_INCREASE  # Mean return on long-term investment
    )
    short_term_investment_std_return = (
        0.01  # Standard deviation of return on long-term investment
    )

    # Loop over time periods
    for t in range(TIME_PERIODS):
        # Compute TFP for each firm at time t
        firm_tfp[:, t] = firm_productivity * (
            1
            + firm_research_investment[:, 0] * SHORT_TERM_TFP_INCREASE
            + firm_research_investment[:, 1] * LONG_TERM_TFP_INCREASE
        )

        # Compute investment choices for each firm at time t

        # Availabilty
        # long term investiment
        long_term_investment_return = np.random.binomial(
            n=1, p=LONG_TERM_INVESTIMENT_PROB, size=NUM_FIRMS
        )

        # Return
        # short term investiment return
        SHORT_TERM_TFP_INCREASE = np.random.normal(
            short_term_investment_mean_return,
            short_term_investment_std_return,
            NUM_FIRMS,
        )

        # Return of the long term increse
        LONG_TERM_TFP_INCREASE = np.random.normal(
            long_term_investment_mean_return, long_term_investment_std_return, NUM_FIRMS
        )

        num_long_term_investors = 0
        num_short_term_investors = 0
        for i in range(NUM_FIRMS):
            # Compute expected returns for short and long term investments
            if (
                firm_research_investment[i, 2] == 0
            ):  # if long-term investment finished, recalculate expected returns
                short_term_expected_return = (
                    firm_tfp[i, t] * SHORT_TERM_TFP_INCREASE[i]
                    - SHORT_TERM_INVESTIMENT_COST
                )
                long_term_expected_return = (
                    firm_tfp[i, t] * LONG_TERM_TFP_INCREASE[i]
                    - LONG_TERM_INVESTIMENT_COST
                )
            else:  # if long-term investment ongoing, expected return is zero
                short_term_expected_return = (
                    firm_tfp[i, t] * SHORT_TERM_TFP_INCREASE[i]
                    - SHORT_TERM_INVESTIMENT_COST
                )
                long_term_expected_return = 0

            # Choose investment with highest expected return
            if long_term_expected_return > short_term_expected_return:
                firm_research_investment[i, 1] += 1
                firm_research_investment[i, 0] = 0
                firm_research_investment[i, 2] = LONG_TERM_INVESTIMENT_TIME
                num_long_term_investors += 1
            else:
                firm_research_investment[i, 0] += 1
                num_short_term_investors += 1

            # If long-term investment was made, adjust investment counter
            if (
                long_term_investment_return[i] == 1
                and firm_research_investment[i, 1] == 1
            ):
                firm_research_investment[i, 2] = LONG_TERM_INVESTIMENT_TIME - 1
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
    group_time = aggregate_data.groupby(["Time"])["Num_Long_Term_Investors"].max()
    share_of_long = group_time / NUM_FIRMS

    plt.plot(time, share_of_long)
    plt.xlabel("Time")
    plt.ylabel("Share of long")
    plt.show()

    # Plot firm TFP over time
    for i in range(NUM_FIRMS):
        plt.plot(time, firm_tfp[i])
    plt.title("Firm TFP over Time")
    plt.xlabel("Time")
    plt.ylabel("TFP")
    plt.show()
    return (aggregate_data, firm_tfp)


# Define parameters
TIME_PERIODS = 100
NUM_FIRMS = 3
SHORT_TERM_INVESTIMENT_COST = 0.01
LONG_TERM_INVESTIMENT_COST = 0.05
SHORT_TERM_TFP_INCREASE = 0.02
LONG_TERM_TFP_INCREASE = 0.1
LONG_TERM_INVESTIMENT_TIME = 3
LONG_TERM_INVESTIMENT_PROB = 0.8

# Simulate model
df, firm_tfp = model(
    TIME_PERIODS,
    NUM_FIRMS,
    SHORT_TERM_INVESTIMENT_COST,
    LONG_TERM_INVESTIMENT_COST,
    SHORT_TERM_TFP_INCREASE,
    LONG_TERM_TFP_INCREASE,
    LONG_TERM_INVESTIMENT_TIME,
    LONG_TERM_INVESTIMENT_PROB,
)


def marginal_revenue(demand_function, market_quantity, price):
    # Calculate the derivative of the demand function at the current market quantity
    h = 0.001
    derivative = (
        demand_function(market_quantity + h) - demand_function(market_quantity)
    ) / (h)

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


def calc_marginal_cost(tfp):
    # Define non-linear cost function
    def cost_func(q, tfp):
        return np.exp(tfp) * q

    # Calculate marginal cost
    q = 1  # Set quantity to 1
    cost = cost_func(q, tfp)
    q_increment = 0.0001  # Set small increment for numerical differentiation
    cost_increment = cost_func(q + q_increment, tfp) - cost
    marginal_cost = cost_increment / q_increment

    return marginal_cost


def simulate_market(firm_tfp, period):
    # Set the number of firms
    NUM_FIRMS = firm_tfp.shape[0]

    # Set the initial price and quantities for each firm
    price = 10
    quantities = np.ones(NUM_FIRMS) * market_demand(price) / NUM_FIRMS
    market_power = np.zeros(NUM_FIRMS)

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
                "Marginal cost",
                "Marginal revenue",
                "Profit",
                "Quantity",
                "Price",
                "Market power",
            ]
        )

        # Update the quantities for each firm based on the new price

        for j in range(NUM_FIRMS):
            # Calculate the marginal revenue for firm j
            marginal_revenue_j = marginal_revenue(
                demand_function, total_quantity, price
            )

            # Calculate the marginal cost for firm j
            marginal_cost = calc_marginal_cost(firm_tfp[j])

            # Update the quantity for firm j based on the marginal revenue and marginal cost

            if marginal_revenue_j > marginal_cost:
                quantities[j] += 1
            elif marginal_revenue_j < marginal_cost:
                quantities[j] -= 1

            # Calculate the profit for firm j
            revenue = quantities[j] * price
            cost = quantities[j] * marginal_cost
            profit = revenue - cost

            # Calculate the market power for firm j
            market_power[j] = quantities[j] / total_quantity

            # Append results to dataframe
            new_row = pd.DataFrame(
                {
                    "Time": period,
                    "Firm": j,
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
        results = pd.DataFrame(new_rows.tail(NUM_FIRMS))

    return results


market_power = np.zeros(NUM_FIRMS)

results_list = []
for t in range(TIME_PERIODS):
    firm_tfp_period = firm_tfp[:, t]
    period_results = simulate_market(firm_tfp_period, t)
    results_list.append(period_results)

results = pd.concat(results_list, ignore_index=True)
print(results)
