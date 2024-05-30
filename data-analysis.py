import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from nltk.stem import PorterStemmer
import text_hammer as th
from nltk.corpus import stopwords
import re
import math
# Load your dataset into a pandas dataframe
df = pd.read_csv('dataset.csv')
stop_words = set(stopwords.words("english"))
tqdm.pandas()
ps = PorterStemmer()

def remove_stopwords(text):
    try:
        return " ".join([word for word in text.split() if word not in stop_words])
    except AttributeError:
        print(text)

def text_preprocessing(df,col_name):
    column = col_name
    df[column] = df[column].progress_apply(lambda x: x.lower())
    #df[column] = df[column].progress_apply(lambda x: th.cont_exp(x)) #you're -> you are; i'm -> i am
    df[column] = df[column].progress_apply(lambda x: remove_stopwords(x))
    #df[column] = df[column].progress_apply(lambda x: th.remove_common_words(x, ['allow']))#, 'user', 'allow', 'service', 'version', 'file', 'attacker', 'arbitrary', 'cause', 'execute', 'denial']))
    df[column] = df[column].progress_apply(lambda x: ps.stem(x))
    #df[column] = df[column].progress_apply(lambda x: lemmatizing(x))
    #df[column] = df[column].progress_apply(lambda x: th.remove_special_chars(x))
    
    #df[column] = df[column].progress_apply(lambda x: th.make_base(x)) #ran -> run,
    df[column] = df[column].progress_apply(lambda x: re.sub(r'\d+', '', x))  # remove numbers
    #df[column] = df[column].progress_apply(lambda x: re.sub(r'[^\w\s//-]', '', x))
    return(df)

df = text_preprocessing(df, 'description')
# Assume your dataset has two columns: 'text' and 'category'
# where 'category' is either 0 or 1

# Tokenize the text data
tokenized_data = df['description'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokenized_data = tokenized_data.apply(lambda x: [word for word in x if word not in stop_words])

# Create a dictionary to store the results
results = {}

# Most common words in each category
for category in ['esv', 'gpsv']:
    category_data = tokenized_data[df['label'] == category]
    word_counts = Counter(word for tokens in category_data for word in tokens)
    results[f'most_common_{category}'] = word_counts.most_common(20)
#print(results)

# Word frequencies in each category
for category in ['esv', 'gpsv']:
    category_data = tokenized_data[df['label'] == category]
    word_freqs = {}
    for tokens in category_data:
        for word in tokens:
            #print(word)
            word_freqs[word] = word_freqs.get(word, 0) + 1
    #print(word_freqs)
    results[f'word_freqs_{category}'] = word_freqs
    print(results)

# Cross words between categories
cross_words = set()
for category in ['esv', 'gpsv']:
    category_data = tokenized_data[df['label'] == category]
    for tokens in category_data:
        cross_words.update(tokens)
results['cross_words'] = list(cross_words)

# Unique words for each category
unique_words = {}
for category in ['esv', 'gpsv']:
    category_data = tokenized_data[df['label'] == category]
    unique_words[category] = set(word for tokens in category_data for word in tokens) - cross_words
results['unique_words'] = unique_words

# Plot the most common words in each category
#for category in ['esv', 'gpsv']:
#    word_counts = results[f'most_common_{category}']
#    plt.bar(range(len(word_counts)), [count for word, count in word_counts])
#    plt.xlabel('Word Index')
#    plt.ylabel('Frequency')
#    plt.title(f'Most Common Words in Category {category}')
#    plt.show()
print("ksljfwojfwofjwofejowfejwofejwoefjwofjweofjwofjwofiwefjwef")
# Print the results
print(results)