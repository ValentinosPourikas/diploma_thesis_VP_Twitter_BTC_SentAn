#%%
"""
Sentiment Analysis and Labeling

This script performs sentiment analysis on preprocessed tweet data using VADER and TextBlob sentiment analyzers.
It loads data from a JSON file, labels the sentiment of each tweet, counts sentiment categories, and saves the labeled data.

Usage:
- Ensure 'preprocessed_tweet_data.json' is available in the same directory.
- Run this script to perform sentiment analysis and save the labeled data as 'labeled_data.json'.
"""
# Total number of posts classified as "neu" by VADER: 5874
# Total number of pos posts: 5787
# Total number of neg posts: 2883
# Total number of neu posts: 4436
# Total number of posts: 13106
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import json

# Load data from a JSON file
def load_data(filename):
    '''Load data from a JSON file.'''
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

# Label sentiment using VADER
def label_sentiment_vader(text, analyzer):
    """
    Label sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    Parameters:
    - text (str): The text to analyze.
    - analyzer (SentimentIntensityAnalyzer): The VADER sentiment analyzer.

    Returns:
    - str: The sentiment label ('pos', 'neg', or 'neu').
    """
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return 'pos' if compound_score > 0 else ('neg' if compound_score < 0 else 'neu')

# Label sentiment using TextBlob
def label_sentiment_textblob(text):
    """
    Label sentiment using TextBlob.

    Parameters:
    - text (str): The text to analyze.

    Returns:
    - str: The sentiment label ('pos', 'neg', or 'neu').
    """
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    return 'pos' if sentiment_score > 0 else ('neg' if sentiment_score < 0 else 'neu')

# Count sentiments in a list of entries
def count_sentiments(entries):
    """
    Count sentiments in a list of entries.

    Parameters:
    - entries (list): A list of entries where each entry contains a sentiment label.

    Returns:
    - dict: A dictionary containing sentiment category counts.
    """
    counts = {'pos': 0, 'neg': 0, 'neu': 0}
    for entry in entries:
        sentiment_label = entry[-1]
        counts[sentiment_label] += 1
    return counts

def process_data(input_filename, output_filename):
    """
    Process and label data, and save the results to a file.

    Parameters:
    - input_filename (str): The name of the input JSON file.
    - output_filename (str): The name of the output JSON file for labeled data.

    Returns:
    - None
    """
    # Load data
    preprocessed_tweet_data = load_data(input_filename)
    
    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Create a list to store the labeled entries
    labeled_entries = []
    neu_by_vader=0
    # Label each entry using VADER and TextBlob as needed, and store it in labeled_entries
    for entry in preprocessed_tweet_data:
        username, date, post_time, tweet_text = entry
        sentiment_label_vader = label_sentiment_vader(tweet_text, analyzer)
        
        # Check if VADER labels the post as non-neutral (pos or neg)
        if sentiment_label_vader in ('pos', 'neg'):
            labeled_entry = [username, date, post_time, tweet_text, sentiment_label_vader]
        else:
            # If VADER labels as neu, then perform TextBlob analysis
            neu_by_vader=neu_by_vader+1
            sentiment_label_textblob = label_sentiment_textblob(tweet_text)
            labeled_entry = [username, date, post_time, tweet_text, sentiment_label_textblob]
        
        labeled_entries.append(labeled_entry)
    print('Total number of posts classified as "neu" by VADER:' , neu_by_vader)
    # Count sentiments
    sentiment_counts = count_sentiments(labeled_entries)
    
    # Print the counts for each category
    total_posts=0
    for sentiment, count in sentiment_counts.items():
        print(f"Total number of {sentiment} posts:", count)
        total_posts=total_posts+count
    print("Total number of posts:", total_posts)
    
    # Save the labeled entries to a new file
    try:
        with open(output_filename, 'w') as file:
            json.dump(labeled_entries, file)
        # print(f"Labeled entries saved to {output_filename}.")
    except IOError as e:
        print(f"Error saving labeled entries: {e}")

if __name__ == "__main__":
    input_filename = 'preprocessed_tweet_data.json'
    output_filename = 'labeled_data.json'
    process_data(input_filename, output_filename)

#Manual definition of tweet posts
# def label_sentiment(text):
#     sentiment = input(f"Define sentiment for tweet's text: {text}\nEnter 1 for negative, 2 for neutral, or 3 for positive: ")
#     if sentiment == '1':
#         return 'neg'
#     elif sentiment == '2':
#         return 'neu'
#     elif sentiment == '3':
#         return 'pos'
#     else:
#         return 'neu'  # Default to neutral if input is not recognized

# %%
