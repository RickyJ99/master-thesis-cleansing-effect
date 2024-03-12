# difference  under no monitoring costs
import pandas as pd
import numpy as np

# Define the paths to the two CSV files
file_path_1 = "output_data/exit_friction_stats.csv"
file_path_2 = "output_data/noexit_nofriction_stats.csv"

# Read the CSV files into pandas DataFrames
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

# Ensure both DataFrames have the same length
if len(df1) == len(df2):
    # Calculate the log difference between the 'Total_Production_actual' columns (mean)
    df_log_diff_mean = round(
        np.log(
            df1["Total_Production_actual_mean"] / df2["Total_Production_actual_mean"]
        ),4
    )

    # Calculate the log differences for CI upper and lower bounds
    # For log differences, the bounds calculation needs adjustment to reflect the percentage change interpretation
    df_log_diff_ci_upper = round(
        np.log(
            df1["Total_Production_actual_CI_Upper"]
            / df2["Total_Production_actual_CI_Upper"]
        ),
        4,
    )
    df_log_diff_ci_lower = round(
        np.log(
            df1["Total_Production_actual_CI_Lower"]
            / df2["Total_Production_actual_CI_Lower"]
        ),
        4,
    )

    # For SEM, you might want to keep the original calculation or adjust according to your methodological preferences
    # Here we keep the original calculation as an example
    df_diff_sem = round(
        (
            (df1["Total_Production_actual_SEM"] ** 2)
            + (df2["Total_Production_actual_SEM"] ** 2)
        )
        ** 0.5,
        4,
    )

    # Create a new DataFrame to store the result including log diff mean, CI bounds, and SEM
    df_result = pd.DataFrame(
        {
            "Step": df1["Step"],
            "Total_Production_actual_log_diff_mean": df_log_diff_mean,
            "Total_Production_actual_log_diff_CI_Upper": df_log_diff_ci_upper,
            "Total_Production_actual_log_diff_CI_Lower": df_log_diff_ci_lower,
            "Total_Production_actual_diff_SEM": df_diff_sem,  # Kept as original; consider adjustments for log interpretation
        }
    )

    # Save the result to a new CSV file
    result_path = "output_data/diff_fri.csv"
    df_result = df_result[df_result["Step"] != 0]
    df_result.to_csv(result_path, index=False)

    print(f"Result saved to {result_path}")
else:
    print("Error: The DataFrames have different lengths.")
