#%%
"""
This script processes labeled data containing tweets for sentiment analysis. It splits the data into training and testing sets, and then converts and saves them as JSON files. The processed JSON files include tweet text, sentiment labels, and dates associated with each tweet.

Usage:
    - Ensure 'labeled_data.json' containing the labeled tweet data is present in the same directory.
    - Run this script to split and process the data into 'train_data.json' and 'test_data.json' files.
"""
import json
from sklearn.model_selection import train_test_split

def process_data(X, y, data, output_filename):
    """
    Process data and save it as a JSON file.

    Args:
        X (list): List of tweet texts.
        y (list): List of sentiment labels.
        data (list): List of data records.
        output_filename (str): Name of the output JSON file.

    Returns:
        None
    """
    # Create a list of dictionaries for each tweet
    json_data = [
        {
            "text": tweet_text,
            "sentiment": sentiment,
            "date": tweet_record[1]
        }
        for tweet_text, sentiment, tweet_record in zip(X, y, data)
    ]

    # Serialize the list of dictionaries to a JSON file
    with open(output_filename, 'w') as file:
        json.dump(json_data, file)

def main():
    # Read the labeled data from the file
    with open('labeled_data.json', 'r') as file:
        labeled_data = json.load(file)

    # Split data into training and testing sets using stratified sampling
    X = [tweet[3] for tweet in labeled_data]  # tweet[3] contains the text of the tweet
    y = [tweet[4] for tweet in labeled_data]  # tweet[4] contains the sentiment of the tweet
    X_train, X_test, y_train, y_test, train_data, test_data = train_test_split(
        X, y, labeled_data, test_size=0.2, stratify=y, random_state=42)

    # Process and save training and test data
    process_data(X_train, y_train, train_data, 'train_data.json')
    process_data(X_test, y_test, test_data, 'test_data.json')
    
    sentiment_counts = {}
    for sentiment in y_train:
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
        else:
            sentiment_counts[sentiment] = 1

    # Print the sentiment counts
    print("Sentiment counts for the train set")
    print("--------------------------------")   
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment} : {count} ({round(count/len(y_train)*100,2)} % )")

if __name__ == "__main__":
    main()
# %%
