#%%
"""
Temporal Analysis of Influencer Posts

This code performs a temporal analysis of the collected influencer posts from the 'original_data.json' file.
It extracts the dates of each post, converts them into datetime objects, and calculates the range of dates
spanned by these posts. The analysis aids in understanding the timeframe over which influencer posts were made.

The code reads the data from the JSON file, processes the dates, and prints the earliest and latest dates among
the influencer posts. This information provides insights into the temporal scope of the influencer content.

Result:
First date: 2010-12-07
Last date: 2019-11-23
"""

from datetime import datetime
import json

# Read the original_data from the file
with open('original_data.json', 'r') as file:
    original_data = json.load(file)

# Extract dates and convert them to datetime objects
all_dates = [datetime.strptime(sublist[1], "%Y-%m-%d") for sublist in original_data]

# Find the first and last dates
first_date = min(all_dates)
last_date = max(all_dates)

# Print the results
print("First date:", first_date.strftime('%Y-%m-%d'))
print("Last date:", last_date.strftime('%Y-%m-%d'))
####
# >>>
#First date: 2010-12-07
#Last date: 2019-11-23

