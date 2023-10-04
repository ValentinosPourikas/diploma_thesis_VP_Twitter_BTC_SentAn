#%%
# Association Analysis for Twitter Sentiments and Effect Variables
"""
This script performs association analysis between Twitter sentiment labels ('pos', 'neg', 'neu')
and different effect variables representing durations. It calculates Chi-Square statistics,
p-values, Cramér's V, and Contingency Coefficients to assess the associations.

Usage:
1. Ensure that the required dependencies are installed using 'pip install pandas scipy numpy matplotlib seaborn'.

2. Prepare the input data:
   - The script expects a JSON file named 'joined_with_differences.json' containing data with
     sentiment labels ('pos', 'neg', 'neu') and effect variables representing different durations.

3. Customize the script by adjusting parameters and settings as needed.

4. Run the script to:
   - Calculate association statistics for each effect variable.
   - Display Chi-Square statistics, p-values, Cramér's V, and Contingency Coefficients.
   - Visualize the results using bar plots.

Dependencies:
- pandas: For data manipulation and handling.
- scipy: For statistical tests and calculations.
- numpy: For numerical operations.
- matplotlib: For data visualization.
- seaborn: For creating bar plots.

Input:
- 'joined_with_differences.json': JSON file containing data with sentiment labels and effect variables.

Output:
- Bar plots displaying Chi-Square statistics, p-values, Cramér's V, and Contingency Coefficients for each duration.

Author: Valentinos Pourikas
Date: September 2023
"""



import json
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_chi_square_stats(sentiment, effect):
    # Create a contingency table
    contingency_table = pd.crosstab(sentiment, effect)
    
    # Perform Chi-Square test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Calculate Cramér's V
    n = len(sentiment)
    min_dim = min(contingency_table.shape)
    cramer_v = np.sqrt(chi2 / (n * (min_dim - 1)))
    
    # Calculate Contingency Coefficient (C)
    C = np.sqrt(chi2 / (chi2 + n))
    
    return chi2, p, cramer_v, C

# Load your data
with open('joined_with_differences.json', 'r') as file:
    joined_with_differences = json.load(file)

# Define the effect variables and durations
sentiment = [entry[1] for entry in joined_with_differences]
effect_variables = [
    [entry[2] for entry in joined_with_differences],  # 1-Day Effect
    [entry[3] for entry in joined_with_differences],  # 2-Day Effect
    [entry[4] for entry in joined_with_differences],  # 3-Day Effect
    [entry[5] for entry in joined_with_differences],  # 4-Day Effect
    [entry[6] for entry in joined_with_differences],  # 5-Day Effect
    [entry[7] for entry in joined_with_differences],  # 6-Day Effect
    [entry[8] for entry in joined_with_differences],  # 7-Day Effect
    [entry[9] for entry in joined_with_differences],  # 8-Day Effect

]
effect_durations = ["1-Day", "2-Day", "3-Day", "4-Day", "5-Day", "6-Day", "7-Day","8-Day"]
#%%
# Initialize lists to store the statistic values
chi2_statistics = []
p_values = []
cramer_v_values = []
contingency_c_values = []

# Iterate over effect durations
for i, effect in enumerate(effect_variables):
    chi2, p, cramer_v, C = calculate_chi_square_stats(sentiment, effect)
    
    # Append the values to the respective lists
    chi2_statistics.append(chi2)
    p_values.append(p)
    cramer_v_values.append(cramer_v)
    contingency_c_values.append(C)
    
    
    #Print the contingency table here
    if i == 0:
        contingency_table_1_day = pd.crosstab(sentiment, effect_variables[i])
        print("Contingency Table for 1-Day Effect:")
        print(contingency_table_1_day)
        
        # Get the row and column totals
        row_totals = contingency_table_1_day.sum(axis=1)
        column_totals = contingency_table_1_day.sum(axis=0)

        # Calculate the grand total
        grand_total = contingency_table_1_day.values.sum()

        # Initialize a new DataFrame for expected values with the same index and columns as the contingency table
        expected_values_1_day = pd.DataFrame(index=contingency_table_1_day.index, columns=contingency_table_1_day.columns)

        # Calculate expected values for each cell
        for ik in range(len(row_totals)):
            for j in range(len(column_totals)):
                expected_value = (row_totals[ik] * column_totals[j]) / grand_total
                expected_values_1_day.iloc[ik, j] = expected_value

        # Print the expected values
        print("Expected Values for 1-Day Effect:")
        print(expected_values_1_day,"\n")




    
    # Print the statistics for the current day
    print(f"Statistics for {effect_durations[i]}:")
    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Cramér's V: {cramer_v}")
    print(f"Contingency Coefficient (C): {C}")
    print("\n")

# Create separate plots for each statistic
plt.figure(figsize=(12, 8))

# Plot 1: P-value
plt.subplot(2, 2, 1)
ax = sns.barplot(x=effect_durations, y=p_values, palette="viridis")
ax.set_xlabel("Days", fontsize=12)
ax.set_ylabel("P-value", fontsize=12)
ax.set_title("P-value vs. Days", fontsize=14)
# Modify y-axis labels (double the values)
new_y_ticks = [0.01,0.03,0.05, 0.1,0.2,0.3,0.4,0.5]
ax.set_yticks(new_y_ticks)
plt.grid(True)


# Plot 2: Chi-Square Statistic
plt.subplot(2, 2, 2)
ax = sns.barplot(x=effect_durations, y=chi2_statistics, palette="viridis")
ax.set_xlabel("Days", fontsize=12)
ax.set_ylabel("Chi-Square Statistic", fontsize=12)
ax.set_title("Chi-Square Statistic vs. Days", fontsize=14)
# ax.set_yscale("log")


# Plot 3: Cramér's V
plt.subplot(2, 2, 3)
ax = sns.barplot(x=effect_durations, y=cramer_v_values, palette="viridis")
ax.set_xlabel("Days", fontsize=12)
ax.set_ylabel("Cramér's V", fontsize=12)
ax.set_title("Cramér's V vs. Days", fontsize=14)

# Plot 4: Contingency Coefficient (C)
plt.subplot(2, 2, 4)
ax = sns.barplot(x=effect_durations, y=contingency_c_values, palette="viridis")
ax.set_xlabel("Days", fontsize=12)
ax.set_ylabel("Contingency Coefficient (C)", fontsize=12)
ax.set_title("Contingency Coefficient (C) vs. Days", fontsize=14)

plt.tight_layout()
plt.show()
# %%
