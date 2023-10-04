#%%
'''This Python script, handle_tweets.py, processes tweet data from a CSV file named 'tweets.csv'. It extracts
essential information such as usernames,dates, times,and text content from each tweet. By comparing usernames
with a list of cryptocurrency influencers, it identifies and collects posts from these influencers. The script
then compiles this data into a list and saves it as 'original_data.json'. The collection part is time consuming 
so the execution time is measured and printed, and the number of total posts and influencer posts found is displayed as well.
A random post record in the unprocessed data:  ['1153219347910987776;OneCryptoCap1;OneCryptoCap News;;2019-07-22 08:24:52+00;0;0;0;"Did you catch it? @FishChainGame joined us for a #LOVENEO episode to discuss the launch of #NEOFish', ' a game tailor developed for #NEP5! Check out deta https://t.co/ryFKShfQVh #crypto #cryptocurrency #blockchain #ethereum #btc #bitcoinmining #bitcoins #litecoin"']
total Tweets: 17345316
num_of_influncer_posts: 13533
Elapsed time: 180.6220202445984 seconds.
'''

# Import necessary libraries
import time
import csv
import os
import json
from crypto_influencers import cryptoinfluencers



# Record the start time
start_time = time.time()

# Initialize variables
original_data = []
num_of_rows = 0
num_of_influncer_posts = 0
influencer_usernames = set(cryptoinfluencers)

# Open the 'tweets.csv' file and process its content
with open('tweets.csv', 'r', encoding='utf-8') as btc_file:
    reader = csv.reader(btc_file)
    for row in reader:
        #if num_of_rows < 100000:
        try:
            # Split the row content into individual data elements
            row_of_info = row[0].split(";") #row_of_info is the data of ONE post like : username, 
            username = row_of_info[1]
            # print(username)
            if username in influencer_usernames:
                # Extract relevant data from the row and append to 'original_data'
                dateofpost, timeofpost, textofpost = row_of_info[4].split(" ")[0], row_of_info[4].split(" ")[1], row_of_info[-1] #naming the data that i got from the csv as date,time (separated the datetime) and text
                original_data.append([username, dateofpost, timeofpost, textofpost])
                num_of_influncer_posts += 1
            num_of_rows += 1
        except IndexError:
            continue
    print("total tweets:", num_of_rows)
    print("num_of_influncer_posts:", num_of_influncer_posts)
    
# print(original_data)

# Save the collected data to 'original_data.json' file
with open('original_data.json', 'w') as file:
    json.dump(original_data, file)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Print the time taken to run the full program
print(f"Elapsed time: {elapsed_time} seconds.")
# %%
