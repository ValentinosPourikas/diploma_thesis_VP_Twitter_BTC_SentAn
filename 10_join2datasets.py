#%%
'''
This script processes and joins two datasets to create a new dataset with extended information.
The resulting dataset includes data from influencer posts and corresponding Bitcoin market data.
For each date, the script extracts the date and matches it with Bitcoin market data.
The script then appends various data fields from both datasets, including mean_OpenClose values for multiple days after each post.
The file joined_2datasets_with_nextdays_mean_OC.json has fields date,sentiment, close,open,high,low,vol,change%,meanHighLow,meanOpenClose,nextday_mean_OpenClose,mean_OpenCloseof2daysafter,..,mean_OpenCloseof8daysafter
'''
import json
from datetime import datetime, timedelta  # Import timedelta

# Read the .json file
with open("daily_sentiment_results.json", "r") as file:
    daily_sentiments = json.load(file)

# Read the BTC_values_augmented.json file
with open("BTC_values_augmented.json", "r") as file:
    btc_values = json.load(file)

# Convert the dates in BTC_values to datetime objects for comparison 
# Uses the unique datetime object keys to create a dictionary with values all its' data.
btc_data_by_date = {datetime.strptime(date_str, "%Y-%m-%d"): data for date_str, data in btc_values.items()}

# Create a new list to store the joined data
joined_data = []
UNTIL_NUM_DAYS_AFTER = 8

# Iterate through the entries in the peprocessed_labeled_data list
for entry in daily_sentiments:
    date=entry[0]
    date = datetime.strptime(date, "%Y-%m-%d")  # Convert the date to a datetime object
    if date in btc_data_by_date:  # Check if the date exists in the btc_data_by_date dictionary
        btc_data = btc_data_by_date[date]  # Get the BTC data for the corresponding date
        joined_entry = entry + list(btc_data.values())  # Combine the entry with the BTC data of the same date. 
        #At this step we only care about the date of the post and not the time of it. The time of the post isnt used at all.
        # Append values for each day from the current date till the next UNTIL_NUM_DAYS_AFTER days
        for days_after in range(1, UNTIL_NUM_DAYS_AFTER + 1): #we want to add to the row of data every btc meanOpenClose value from a day after until 7 days after.
            next_date = date + timedelta(days=days_after)
            # Use a list comprehension to append mean_OpenClose values to the joined entry
            next_btc_data = btc_data_by_date.get(next_date)
            if next_btc_data:
                joined_entry.extend([next_btc_data["mean_OpenClose"]])
            else: print(f"Error. The date {next_date} doesn't exist in the BTC values archive")
        joined_data.append(joined_entry)
    else : # Use a list comprehension to append mean_OpenClose values to the joined entry
            next_btc_data = btc_data_by_date.get(next_date)
            if next_btc_data:
                joined_entry.extend([next_btc_data["mean_OpenClose"]])
            else:
                print(f"Error. The date {next_date} doesn't exist in the BTC values archive")

with open("joined_2datasets_with_nextdays_mean_OC.json", "w") as file:
    json.dump(joined_data, file)
    
