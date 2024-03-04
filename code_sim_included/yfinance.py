import yfinance as yf
import pandas as pd
import numpy as np


# Function to fetch historical stock price data and calculate returns
def get_returns(stock_symbols, start_date, end_date):
    data = yf.download(stock_symbols, start=start_date, end=end_date)
    adj_close = data["Adj Close"]
    returns = adj_close.pct_change().dropna()  # Calculate daily returns

    return returns


# Input the list of stock symbols (ticker symbols)
stock_symbols = input(
    "Enter a list of stock symbols (comma-separated, e.g., AAPL,MSFT,GOOGL): "
).split(",")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")

# Fetch historical stock price data and calculate returns
returns = get_returns(stock_symbols, start_date, end_date)

# Print the returns
print("\nReturns:")
print(returns)

# Calculate optimal portfolio weights
risk_free_rate = float(
    input("Enter the risk-free rate (as a decimal, e.g., 0.03 for 3%): ")
)
n = len(stock_symbols)
w = cp.Variable(n)
expected_return = returns.mean() * 252  # Annualize daily returns
risk = cp.quad_form(w, np.cov(returns, rowvar=False) * 252)
objective = cp.Maximize(expected_return - risk_free_rate * risk)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights = w.value

# Print the optimal portfolio weights
print("\nOptimal Portfolio Weights:")
for i in range(len(stock_symbols)):
    print(f"{stock_symbols[i]}: {optimal_weights[i]:.4f}")
