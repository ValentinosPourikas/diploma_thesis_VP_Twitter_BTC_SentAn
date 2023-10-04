
#%% 
# fall/rise/same
#joined_2datasets_with_nextdays)mean_OC.json has entries with fields:
# date(0),sentiment(1), close(2),open(3),high(3),low(5),vol(6),change%(7),meanHighLow(8),meanOpenClose(9),nextday_mean_OpenClose(10),mean_OpenCloseof2daysafter(11),..,mean_OpenCloseof7daysafter(16),mean_OpenCloseof8daysafter(17)
"""
Analyzes historical Bitcoin data to determine whether the price rose, fell, or remained the same in the days following a given date. It calculates these labels based on predefined thresholds for price changes.

Data Input:
- The script expects a JSON file named 'joined_2datasets_with_nextdays_mean_OC.json' to be present in the working directory. This JSON file should contain historical Bitcoin data with fields including date, sentiment, close price, open price, high price, low price, volume, percentage change, mean high-low, mean open-close, and mean open-close values for various days in the future.

Data Output:
- The script generates a new JSON file named 'joined_with_differences.json' that includes the original data along with labels indicating whether the Bitcoin price rose, fell, or remained the same in the days following each entry.

Note:
- Ensure that the input JSON file ('joined_2datasets_with_nextdays_mean_OC.json') is located in the same directory as this script before running.

"""

import json
from collections import Counter
import subprocess



DAYS_AFTER = 8

def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def calculate_labels(values, thresholds):
    # Extract the meanOpenClose value of Bitcoin for this day
    field_of_meanOC=7
    meanOC = values[field_of_meanOC]  # The 8th value of the list "values" is the meanOpenClose value of Bitcoin for this day
    # Calculate the differences between meanOC and mean_open_close values for each day after
    differences = []
    for i in range(1,DAYS_AFTER+1):
        # Calculate the difference for the i-th day after
        #8,9,10,..,15 are the positions of the meanOC Bitcoin values of 1,2,3,..,8 days after
        mean_open_close_after_i_days = values[field_of_meanOC + i]
        difference = round(mean_open_close_after_i_days - meanOC, 2)
        differences.append(difference)
    # Determine the labels based on thresholds
    labels = []
    for diff in differences:
        if diff > thresholds['rise']:
            labels.append('rise')
        elif diff < thresholds['fall']:
            labels.append('fall')
        else:
            labels.append('same')
    return labels

def main():
    joined_with_nextdays_mean_OC = load_data('joined_2datasets_with_nextdays_mean_OC.json')
    
    thresholds = {'rise': 25, 'fall': -25} #a dictionary for thresholds

    modified_data = [] # list to store the modified entries
    contingency_tables={}
    # Initialize a dictionary to store the contingency table counts
    contingency_table = {'pos': {'rise': 0, 'same': 0, 'fall': 0},
                         'neu': {'rise': 0, 'same': 0, 'fall': 0},
                         'neg': {'rise': 0, 'same': 0, 'fall': 0}}
    
    # Iterate through the entries in the data
    for entry in joined_with_nextdays_mean_OC:
        date, sentiment, *daily_values = entry  # Unpack the entry
        labels = calculate_labels(daily_values, thresholds)
        modified_entry = [date, sentiment] + labels
        modified_data.append(modified_entry)
        
        # Update the contingency table counts
        # Initialize a dictionary to store contingency tables for different days after
    # Loop through each day after
    for i in range(1, DAYS_AFTER+1):
        # Create a new contingency table for this day after
        contingency_table = {'pos': {'rise': 0, 'same': 0, 'fall': 0},
                            'neu': {'rise': 0, 'same': 0, 'fall': 0},
                            'neg': {'rise': 0, 'same': 0, 'fall': 0}}

        # Loop through modified_data and update the contingency table for this day after
        for entry in modified_data:
            sentiment_of_day, BTC_label = entry[1], entry[i + 1]  # i + 2 to get the effect on i days after
            contingency_table[sentiment_of_day][BTC_label] += 1

        # Store the contingency table in the dictionary with the day after as the key
        contingency_tables[f'{i} days after'] = contingency_table

    # Print the contingency tables for each day after
    for day_after, table in contingency_tables.items():
        print(f"Contingency Table for {day_after}:")
        print("{:<10} {:<10} {:<10} {:<10}".format('', 'rise', 'same', 'fall'))
        for sentiment, values in table.items():
            print("{:<10} {:<10} {:<10} {:<10}".format(sentiment, values['rise'], values['same'], values['fall']))
        print()
    # Save the modified data to a new JSON file
    with open('joined_with_differences.json', 'w') as file:
        json.dump(modified_data, file)

if __name__ == "__main__":
    main()
    
