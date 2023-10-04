# 

# %% #Code that uses Naive Bayes algorithm to create a pipeline for sentiment analysis of twitter posts
# Total train data:  10484
# Total test data:  2622
"""
This script uses the Naive Bayes algorithm to perform sentiment analysis on Twitter posts.
It includes data preprocessing, model training, evaluation, and the generation of word clouds
for different sentiment categories.

Usage:
1. Ensure that the required dependencies are installed using 'pip install nltk wordcloud matplotlib seaborn scikit-learn'.

2. Prepare the input data:
   - The script expects training and test data in JSON format with fields 'text' for post text
     and 'sentiment' for sentiment labels ('pos', 'neg', or 'neu').

3. Customize the script by adjusting parameters and settings as needed.

4. Run the script to:
   - Train a Naive Bayes model for sentiment analysis.
   - Evaluate the model's accuracy and generate a confusion matrix.
   - Create and display word clouds for each sentiment category ('pos', 'neg', 'neu').
   - Save the predicted sentiments for test data in a JSON file ('predicted_sentiments.json').

Dependencies:
- nltk: Natural Language Toolkit for text processing.
- wordcloud: For generating word clouds.
- matplotlib: For visualizing word clouds and results.
- seaborn: For creating heatmaps in the confusion matrix.
- scikit-learn: For machine learning model training and evaluation.

Input:
- 'train_data.json': JSON file containing training data with 'text' and 'sentiment' fields.
- 'test_data.json': JSON file containing test data with 'text' and 'sentiment' fields.

Output:
- Confusion matrix heatmap.
- Word clouds for 'pos', 'neg', and 'neu' sentiment categories.
- 'predicted_sentiments.json': JSON file containing predicted sentiments for test data.

Author: Valentinos Pourikas
Date: September 2023
"""
#importing 
import json
import nltk
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import GridSearchCV
from tweet_data_prep import custom_tokenizer,negation_words
from matplotlib.ticker import ScalarFormatter
from sklearn.feature_extraction.text import TfidfVectorizer 
#The TfidfVectorizer is a tool that helps convert a collection of text documents into a numerical representation called a vector. It assigns weights to words based on their frequency in a document and their rarity across all documents, capturing their importance in distinguishing documents from each other.
#"TF-IDF" stands for "Term Frequency-Inverse Document Frequency."
#The TF part represents the frequency of a term (word) in a document, while the IDF part represents the inverse document frequency, which measures how rare or common a term is across the entire document collection
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
#The make_pipeline function in scikit-learn (sklearn) allows you to create a pipeline object that sequentially applies a series of data transformations followed by an estimator.
# It simplifies the process of building and applying machine learning pipelines by automatically assigning names to the intermediate steps based on the names of the transformers and estimator provide
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, classification_report
from nltk.corpus import stopwords  # Import stopwords from NLTK corpus
nltk.download('stopwords')
custom_stopwords_list=list(set(stopwords.words('english'))-set(negation_words))
#%%takes too long to run
# Load train and test data from JSON files
with open('train_data.json', 'r') as file:
    train_data = json.load(file)

with open('test_data.json', 'r') as file:
    test_data = json.load(file)

# Separate text, sentiment, and date into separate lists
X_train = [entry['text'] for entry in train_data]
y_train = [entry['sentiment'] for entry in train_data]
X_test = [entry['text'] for entry in test_data]
y_test = [entry['sentiment'] for entry in test_data]


print("Total train data: ", len(X_train)) #the length of the train dataset
print("Total test data: ", len(X_test)) #the length of the test dataset
print("random train data =", X_train[42]) #we can view whats the text of a random post

#%%
min_df=0.00  #0.00
max_df=0.04 # 0.04-1.0 values give the same accuracy
alpha=0.159 #0.159
norm='l2' #l2
maxfeatures=3350 #3350
# Define the range of min_df,max_df,alpha values to test
# min_df_values = np.arange(0.00, 0.03, 0.001).tolist()
# selected_min_df_values = min_df_values[::2]  # Display every 10th value
# max_df_values = np.arange(0.02, 1, 0.02).tolist()
# selected_max_df_values = max_df_values[::5]  # Display every 10th value
# alpha_values = np.arange(0.005,1.0, 0.001).tolist()
# selected_alpha_values = alpha_values[::100] 
# maxfeatures_values = np.arange(1000,11000,50).tolist()
# selected_maxfeatures_values = maxfeatures_values[::10]  
# norm_values=['l1','l2']

# Initialize lists to store min_df values and corresponding accuracies
# min_df_list = []
# max_df_list = []
# norm_list=[]
accuracy_list = []
# alpha_list=[]
# maxfeatures_list=[]

# f1score_list=[]
#
# for min_df in min_df_values:
# for max_df in max_df_values:
# for alpha in alpha_values:
# for maxfeatures in maxfeatures_values:
# for norm in norm_values:
vectorizer=TfidfVectorizer(
    lowercase=True,
    tokenizer=custom_tokenizer,#applies if analyzer == 'word'. Override the bult-in string tokenization.preprocessing and n-grams generation are preserved either way.
    token_pattern=None,
    analyzer='word', #feature should be made of word
    stop_words=custom_stopwords_list, 
    #frequency-based filtering:managing the impact of terms that appear too frequently or too rarely across a corpus.
    min_df=min_df, #tweets are relatively short and we want to capture frequent terms. (0-0.01)--0.01
    max_df=max_df,# relatively short posts.value slightly higher to capture more relevant terms. --
    max_features=maxfeatures, #only consider the top `max_features` ordered by term frequency across the corpus #ignored if vocabulary is not None.--
    norm=norm,#Sum of squares of vector elements is 1.() The cosine similarity between two vectors is their dot product.)--
    use_idf=True, #use_idf=True,smooth_idf=True : Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.
    smooth_idf=True,
    sublinear_tf=False) #sublinear_tf=True if exist very frequent words, can help to reduce their influence and give more weight to less frequent, more informative terms. Might affect accuracy--

multiNB=MultinomialNB(
        alpha=alpha,#A larger alpha results in stronger smoothing, making the probability estimates more uniform. A smaller alpha places more weight on the observed counts in the training data----
        force_alpha=False, #If False and alpha is less than 1e-10, it will set alpha to 1e-10
        fit_prior=True,#this means that if one class is more frequent in the training data, the classifier will take that into account when making predictions.for it to be False, I must have no knowledge of the data.
        )
model = make_pipeline(
    vectorizer, 
    multiNB,
    verbose=False) #initialization of the model and the parameters


model.fit(X_train,y_train) #training of the model with the data and its now outputs
labels =model.predict(X_test) #prediction of the output(target) of the test data. we only use the X_test and not the y_test which indicates the real outputs
probs = model.predict_proba(X_test)
accuracy = np.mean(labels == y_test)
# f1 = f1_score(y_test, labels, average='weighted') # F1-Score: Harmonic mean of precision and recall.Good metric for imbalanced classes.
# min_df_list.append(min_df)
# max_df_list.append(max_df)
# maxfeatures_list.append(maxfeatures)
# alpha_list.append(alpha)
# norm_list.append(norm)
# accuracy_list.append(accuracy)
# f1score_list.append(f1)

# plt.figure()
# Plotting the results
# plt.plot(min_df_list, accuracy_list, marker='o')
# plt.xlabel('min_df')
# plt.plot(max_df_list, accuracy_list, marker='o')
# plt.xlabel('max_df')
# plt.plot(norm_list, accuracy_list, marker='o')
# plt.xlabel('norm')
# plt.plot(alpha_list, accuracy_list, marker='o')
# plt.plot(alpha_list, f1score_list, marker='o')
# plt.xlabel('alpha')
# plt.plot(maxfeatures_list, accuracy_list, marker='o')
# plt.xlabel('max_features')


# plt.ylabel('Accuracy')
# plt.title('Effect of min_df on Accuracy')
# plt.title('Effect of max_df on Accuracy')
# plt.title('Effect of alpha on Accuracy')
# plt.title('Effect of norm on Accuracy')
# plt.title('Effect of max_features on Accuracy')
# plt.grid(True)

## Customize x-axis tick labels to display in decimal format
# plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# plt.xticks(selected_min_df_values, rotation=45)  # Display only the selected subset of tick values
# plt.xticks(selected_max_df_values, rotation=45)  # Display only the selected subset of tick values
# plt.xticks(selected_alpha_values, rotation=45)  # Display only the selected subset of tick values
# plt.xticks(selected_maxfeatures_values, rotation=45)  # Display only the selected subset of tick values


#
# print(f"Accuracy: {accuracy:.4%}")

#  analyze the results to find the best min_df value
# best_min_df = min_df_list[np.argmax(accuracy_list)]
# best_max_df = max_df_list[np.argmax(accuracy_list)]
# best_alpha = alpha_list[np.argmax(accuracy_list)]
# best_norm = norm_list[np.argmax(accuracy_list)]
# best_max_features = maxfeatures_list[np.argmax(accuracy_list)]



# print(f"Best min_df value: {best_min_df}")
# print(f"Best max_df value: {best_max_df}")
# print(f"Best alpha value: {best_alpha}")
# print(f"Best norm value: {best_norm}")
# print(f"Best max_features value: {best_max_features}")

# best_accuracy = max(accuracy_list)
# print(f"Best accuracy on Test set: {best_accuracy:.4%}")
#%%
categories = ['neg', 'neu', 'pos']

conf_mat = confusion_matrix(y_test, labels) #creates a confusion matrix for the output based on what it predicted and whats the real value
print(conf_mat)

sns.heatmap(conf_mat.T, square =True,annot=True,fmt='d', cbar=True,xticklabels=categories,yticklabels=categories)
#a heatmap is a graphical representation of data where the individual values contained in a matrix are represented as colors.
# Heatmaps are commonly used to visualize the distribution and patterns within a dataset 
# they provide a concise visual summary of the data. They allow analysts to quickly identify patterns, trends, and areas of interest within the data

plt.xlabel('true label')
plt.ylabel('predicted label')

accuracy = np.mean(labels == y_test)
precision = precision_score(y_test, labels, average='weighted') # Precision: Proportion of true positive predictions among all positive predictions.how close the measurements are to each other
recall = recall_score(y_test, labels, average='weighted') # Recall: Proportion of true positive predictions among all actual positive instances.
f1 = f1_score(y_test, labels, average='weighted') # F1-Score: Harmonic mean of precision and recall.

# true_negatives = conf_mat[0, 0]
# false_positives = conf_mat[0, 1] + conf_mat[0, 2]  # Sum of false positives for other classes
# false_negatives = conf_mat[1, 0] + conf_mat[2, 0]  # Sum of false negatives for other classes
# true_positives = conf_mat[1, 1] + conf_mat[1, 2] + conf_mat[2, 1] + conf_mat[2, 2]  # Sum of true positives for other classes
# specificity = true_negatives / (true_negatives + false_positives)
# npv = true_negatives / (true_negatives + false_negatives)
# fpr = false_positives / (false_positives + true_negatives)
# fnr = false_negatives / (false_negatives + true_positives)

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
# print(f"Specificity: {specificity:.4f}")
# print(f"Negative Predictive Value (NPV): {npv:.4f}")
# print(f"False Positive Rate (FPR): {fpr:.4f}")
# print(f"False Negative Rate (FNR): {fnr:.4f}")



# # Print the classification report
# print("Classification Report:")
class_report = classification_report(y_test, labels, target_names=categories,digits=4)
print(class_report)

def predict_category(s, model=model):
    pred = model.predict([s])
    return categories[pred[0]]
#%%
def generate_and_display_wordcloud(sentiment_category, tweets):
    # Combine the tweets into a single text
    text = ' '.join(tweets)
    
    # Create the WordCloud object
    wc = WordCloud(
        background_color='white',     # Background color of the word cloud
        max_words=maxfeatures,        # Maximum number of words to display
        stopwords=custom_stopwords_list,  # Custom stopwords to exclude from the word cloud
        width=800, height=400,        # Size of the word cloud image
    )

    # Generate the word cloud
    wordcloud = wc.generate(text)

    # Display the word cloud using Matplotlib
    plt.figure(figsize=(10, 5))  # Set the size of the figure
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off the axis labels
    plt.title(f'Word Cloud for "{sentiment_category}" Category Tweets')
    plt.show()
#%%
# Filter the training data for different sentiment categories
sentiment_categories = ['pos', 'neg', 'neu']
sentiment_tweets = {category: [X_train[i] for i in range(len(X_train)) if y_train[i] == category] for category in sentiment_categories}

# Generate and display word clouds for each sentiment category
for category, tweets in sentiment_tweets.items():
    generate_and_display_wordcloud(category, tweets)
# %% save the results
results = []

for i in range(len(X_test)):
    # Append the data to the 'results' list
    results.append({
        'date': test_data[i]['date'],
        'text': test_data[i]['text'],
        'predicted_sentiment': labels[i] 
    })


# Save the 'results' list to a JSON file
with open('predicted_sentiments.json', 'w') as file:
    json.dump(results, file)
# %%
