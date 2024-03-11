import pandas as pd

# Define the paths to the two CSV files
file_path_1 = "output_data/exit_nofriction_stats.csv"
file_path_2 = "output_data/noexit_nofriction_stats.csv"

# Read the CSV files into pandas DataFrames
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

# Ensure both DataFrames have the same length
if len(df1) == len(df2):
    # Calculate the difference between the 'Total_Production_actual' columns (mean)
    df_diff_mean = df1["Total_Production_actual_mean"] - df2["Total_Production_actual_mean"]
    
    # Calculate the differences for CI upper and lower bounds
    # Note: This calculation assumes you want the diff of bounds themselves, not the recalculation of bounds for the diff
    df_diff_ci_upper = df1["Total_Production_actual_CI_Upper"] - df2["Total_Production_actual_CI_Upper"]
    df_diff_ci_lower = df1["Total_Production_actual_CI_Lower"] - df2["Total_Production_actual_CI_Lower"]
    
    # Calculate the combined SEM for the differences using the formula for the difference between two means
    df_diff_sem = ((df1["Total_Production_actual_SEM"] ** 2) + (df2["Total_Production_actual_SEM"] ** 2)) ** 0.5

    # Create a new DataFrame to store the result including diff mean, CI bounds, and SEM
    df_result = pd.DataFrame({
        'Step': df1['Step'],
        "Total_Production_actual_diff_mean": df_diff_mean,
        "Total_Production_actual_diff_CI_Upper": df_diff_ci_upper,
        "Total_Production_actual_diff_CI_Lower": df_diff_ci_lower,
        "Total_Production_actual_diff_SEM": df_diff_sem,
    })

    # Save the result to a new CSV file
    result_path = "output_data/diff.csv"
    df_result.to_csv(result_path, index=False)

    print(f"Result saved to {result_path}")
else:
    print("Error: The DataFrames have different lengths.")
