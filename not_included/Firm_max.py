import pandas as pd
import numpy as np


def marginal_revenue(demand_function, market_quantity):
    # Calculate the price at the current market quantity
    price = demand_function(market_quantity)

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
    return 2000 - 4 * price


def demand_function(quantity):
    return 500 - 0.25 * quantity


def simulate_market(firm_tfp, period):
    # Set the number of firms
    num_firms = firm_tfp.shape[0]

    # Set the initial price and quantities for each firm
    price = 50
    quantities = np.ones(num_firms) * market_demand(price) / num_firms
    market_power = np.zeros(num_firms)

    # Set the number of iterations for the simulation
    num_iterations = 10000

    # Initialize empty dataframe to store results
    results = pd.DataFrame(
        columns=["Time", "Firm", "Marginal cost", "Profit", "Quantity", "Market power"]
    )

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
                "Profit",
                "Quantity",
                "Market power",
            ]
        )
        # Update the quantities for each firm based on the new price
        for j in range(num_firms):
            # Calculate the marginal revenue for firm j
            marginal_revenue_j = marginal_revenue(demand_function, total_quantity)

            # Calculate the marginal cost for firm j
            marginal_cost = 1 / firm_tfp[j]

            # Update the quantity for firm j based on the marginal revenue and marginal cost
            if marginal_revenue_j > marginal_cost:
                quantities[j] += 1
            elif marginal_revenue_j < marginal_cost:
                quantities[j] -= 1

            # Calculate the profit for firm j
            revenue = quantities[j] * price
            cost = quantities[j] * marginal_cost
            profit = revenue - cost

            # If profit is negative, set quantity to zero
            if profit < 0:
                quantities[j] = 0
                profit = 0

            # Calculate the market power for firm j
            market_power[j] = quantities[j] / total_quantity

            # Append results to dataframe
            new_row = pd.DataFrame(
                {
                    "Time": period,
                    "Firm": j,
                    "Marginal cost": marginal_cost,
                    "Profit": profit,
                    "Quantity": quantities[j],
                    "Market power": market_power[j],
                },
                index=[0],
            )
            new_rows = pd.concat([new_rows, new_row], ignore_index=True)
        results = pd.concat([results, new_rows], ignore_index=True)
    return results


# Set the number of firms and time periods
num_firms = 5
num_periods = 10

# Generate random TFP values for each firm and time period
firm_tfp = np.random.rand(num_firms, num_periods)
market_power = np.zeros(num_firms)

results_list = []
for t in range(num_periods):
    firm_tfp_period = firm_tfp[:, t]
    period_results = simulate_market(firm_tfp_period, t)
    results_list.append(period_results)

results = pd.concat(results_list, ignore_index=True)
print(results)
