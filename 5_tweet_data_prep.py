#%%

"""This code performs several data preprocessing steps on a dataset containing Twitter posts for sentiment analysis. Let's go through each step:

Lowercasing: The code converts all text in the dataset to lowercase. This step is useful as it ensures that words are treated in a case-insensitive manner, preventing the model from considering the same word in different cases as different tokens.
Tokenization: Tokenization is the process of breaking down the text into individual words or tokens.  This step helps in breaking down the text into smaller units, making it easier to analyze and process.
Stopword Removal: The code downloads the set of English stopwords from NLTK and removes them from the tokenized data. Stopwords are common words like "the," "is," "a," etc., which often don't carry much meaning and can be safely removed to reduce noise in the data.
Punctuation Removal: The code uses regular expressions to remove punctuation marks from the tokenized data. Removing punctuation is beneficial as it reduces the number of unique tokens and ensures that words with punctuation are treated the same as their counterparts without punctuation.
Stemming and Lemmatization: The code performs lemmatization on the tokenized data. Stemming reduces words to their base or root form by removing suffixes or prefixes. Lemmatization, on the other hand, reduces words to their base form (lemma) using language-specific rules. For example, both "running" and "ran" will be reduced to "run." By doing both stemming and lemmatization, the code captures different variations of words and reduces sparsity in the data.
Save Preprocessed Data: At various stages of preprocessing, the code saves the intermediate results as separate JSON files (e.g., tokenized_data.json, cleaned_data.json, processed_data_NLTKstem.json)."""
import os
import re
import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import  WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')


# Preprocessing functions
def lower_case(text):
    """Convert text to lowercase."""
    return text.lower()

def custom_tokenizer(text):
    '''The custom_tokenizer function removes tag names, links, and punctuation (except the apostrophe) 
    from the input text and returns a list of cleaned tokens.'''
    
    # Regular expression pattern to match tag names
    tag_pattern = r'@\w+'
    # Regular expression pattern to match links
    link_pattern = r'http\S+'
    
    # Remove tag names from the text
    modified_text = re.sub(tag_pattern, '', text)
    # Remove links from the text
    modified_text = re.sub(link_pattern, '', modified_text)

    # Remove all punctuation except the apostrophe and replace with a whitespace
    punctuation_except_apostrophe = string.punctuation.replace("'", "")
    modified_text = re.sub(r"[{}]".format(re.escape(punctuation_except_apostrophe)), lambda match: ' ', modified_text)
    
    # print(modified_text)
    # Manually tokenize the text
    tokens = modified_text.split(" ")

    # Remove both "'" and " " from the start and end of the strings
    cleaned_list = [word.strip(" '") for word in tokens]

    # Filter out empty strings (delete them)
    tokens = [word for word in cleaned_list if word]

    return tokens


def remove_stopwords(tokens, negation_words_set):
    """Remove stopwords from tokens, excluding negation words."""
    # Access the default English stopwords list
    stopwords_list = stopwords.words('english') 
    #remove terms of btc that give no value to our text
    additional_words_to_remove=['btc','bitcoin','xbt']
    words_to_remove=stopwords_list+additional_words_to_remove
    # Initialize an empty list to store the remaining tokens
    remaining_tokens = []
    
    # Iterate through the tokens and keep only those that are not stopwords or in negation_words_set
    for token in tokens:
        if token not in words_to_remove or token in negation_words_set:
            remaining_tokens.append(token)
    
    return remaining_tokens

def lemmatize_tokens(tokens,lemmatizer):
    """Lemmatize tokens using WordNetLemmatizer."""
    # Perform  lemmatization on each record
    return [lemmatizer.lemmatize(word) for word in tokens]

# List of all negation words and verbs in negation to be excluded during stopword removal
negation_words = [ 'no', 'nor', 'not', "couldn't",'couldn','couldnt',"didn't",'didn','didnt', "doesn't","doesnt","doesn", "hadn't",'hadn','hadnt',
                  "hasn't","hasn","hasnt","haven't","isn't",'isn',"isnt", "mightn't", "mightn","mightnt",
                  "mustn't","mustn", "needn't","needn", "shan't", "shouldn't","shouldn", "wasn't","wasn",
                  "weren't", "weren", "won't", "wouldn't","wouldn", "aren't","aren",'don',"don't"]

if __name__ == "__main__":
    # This code block will only run when the script is executed directly, not when imported as a module in another script.
    RANDOM_POST = 205
    # Read the original_data from the file
    with open('original_data.json', 'r') as file:
        joined_data = json.load(file)

    ####data preparation/preprocessing
    print("Total entries at first:", len(joined_data))

    unique_records = {}  # Create an empty dictionary to store unique records

    for record in joined_data:
        username, date, post_time, tweet_text = record[:4]
        # Use the combination of username, date, post_time, and tweet_text as the key
        record_key = (username, date, post_time, tweet_text)
        if record_key not in unique_records:
            unique_records[record_key] = record  # Store the entire record as the value

    # Convert the unique_records dictionary values to a list of unique records
    unique_records_list = list(unique_records.values())
    print("total unique entries: ",len(unique_records_list))

    preprocessed_data=[]
    lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer outside the loop
    negation_words_set = set(negation_words)  # Convert negation_words to a set for faster lookup

    for post_counter, record in enumerate(unique_records_list):
        # Iterate through the unique joined_data 
        if post_counter==RANDOM_POST: print("original:",record)
        username, date, post_time, tweet_text = record[:4]
        if post_counter==RANDOM_POST: print("text data:",tweet_text)
        ##lowercase 
        lowercased_text = lower_case(tweet_text)
        if post_counter==RANDOM_POST: print("lowercased:",lowercased_text)
        #print("lowercase completed. \n ", len(lowercased_text)

        ##Tokenization
        # Tokenize the text in each record
        tokens = custom_tokenizer(lowercased_text)
        #The custom_tokenizer function removes tag names, links, and punctuation (except the apostrophe) from the input text and returns a list of cleaned tokens.
        ##Print the tokenized data
        if post_counter==RANDOM_POST:print("tokenized:",tokens)
        ## 
        # Stopword removal

        # Stopwords are common words like "the," "is," "a," etc., which often don't carry much meaning and can be safely removed to reduce noise in the data.
        # handling negation words during stopword removal to ensure that negations are not mistakenly discarded
        # Negation words like "not," "don't," "won't," etc., are essential for sentiment analysis
        # because they can completely reverse the sentiment of a sentence or tweet.
        cleaned_tokens = remove_stopwords(tokens, negation_words)

        ##-->possible extra list of words not to be removed from the procedure?

        ##Print the cleaned_tokens with stopwords removed.
        if post_counter==RANDOM_POST:print("stopwords removed:",cleaned_tokens)
            
        ##--> Frequency-based Filtering: Instead of using a fixed list of stopwords, you can perform frequency-based filtering. Identify words that occur too frequently or too infrequently in your dataset and remove them.
        ##--> Words that occur too frequently might be common across all tweets and may not be informative for sentiment analysis. Similarly, words that occur too infrequently might not contribute significantly to sentiment understanding.
        ##--> In our data we dont have words that appear too frequently as observed later.

        ##Stemming and Lemmatization

        ####Sentiment analysis is often about detecting subtle nuances in language and understanding the intent behind the words.
        ###Lemmatization can preserve these nuances better than stemming because it retains actual words that have well-defined meanings.
        ###lemmatization tends to provide a more accurate representation of words while maintaining their context, which can be crucial for understanding sentiment.
        ####That's why stemming is NOT preferable in sentiment analysis.

        lemmatized_tokens = lemmatize_tokens(cleaned_tokens,lemmatizer)
        if post_counter==RANDOM_POST:print("lemmatized:",lemmatized_tokens)
        # Join tokens to create processed text
        processed_text = ' '.join(lemmatized_tokens)
        if post_counter==RANDOM_POST:print("joined to string:",processed_text)
        # Construct processed record
        processed_record = [username, date, post_time, processed_text] + record[4:]
        if post_counter==RANDOM_POST:print("preprocessed record:",processed_record)

        if processed_record[3] != "": preprocessed_data.append(processed_record)
    print("Total unique preprocessed posts (after deleting empty-texted posts): ",len(preprocessed_data))
    with open('preprocessed_tweet_data.json', 'w') as file:
        json.dump(preprocessed_data, file)
        
