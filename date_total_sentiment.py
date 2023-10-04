#%%
import json
from collections import defaultdict

def calculate_sentiment_variable(entries):
    """
    Calculate the sentiment variable based on the entries.

    Args:
        entries (list): List of entries containing text and sentiment.

    Returns:
        int: Sentiment variable (positive - negative count).
    """
    neg_count = sum(1 for _, _, sentiment in entries if sentiment == 'neg')
    pos_count = sum(1 for _, _, sentiment in entries if sentiment == 'pos')
    return pos_count - neg_count

def group_entries_by_date(data):
    """
    Group entries by date.

    Args:
        data (list): List of entries.

    Returns:
        dict: Entries grouped by date.
    """
    entries_by_date = defaultdict(list)
    for entry in data:
        date, text, sentiment = entry['date'], entry['text'], entry['predicted_sentiment']
        entries_by_date[date].append([date, text, sentiment])
    return entries_by_date

def determine_sentiment(sentiment_variable):
    """
    Determine sentiment based on sentiment variable.

    Args:
        sentiment_variable (int): Sentiment variable (positive - negative count).

    Returns:
        str: Sentiment ('pos', 'neg', or 'neu').
    """
    if sentiment_variable > 0:
        return 'pos'
    elif sentiment_variable < 0:
        return 'neg'
    else:
        return 'neu'

def main():
    """
    Main script to calculate and store daily sentiment results.
    """
    with open('predicted_sentiments.json', 'r') as file:
        predicted_sentiments_data = json.load(file)

    entries_by_date = group_entries_by_date(predicted_sentiments_data)
    daily_sentiment_results = []

    negative_count = 0
    neutral_count = 0
    positive_count = 0
    for date, entries in entries_by_date.items():
        sentiment_variable = calculate_sentiment_variable(entries)
        max_sentiment = determine_sentiment(sentiment_variable)
        entry_with_max_sentiment = [date, max_sentiment]
        daily_sentiment_results.append(entry_with_max_sentiment)

        if max_sentiment == 'pos':
            positive_count += 1
        elif max_sentiment == 'neg':
            negative_count += 1
        else:
            neutral_count += 1

    with open('daily_sentiment_results.json', 'w') as file:
        json.dump(daily_sentiment_results, file)

    print(f"Number of days categorized as 'neg': {negative_count}")
    print(f"Number of days categorized as 'neu': {neutral_count}")
    print(f"Number of days categorized as 'pos': {positive_count}")

if __name__ == "__main__":
    main()