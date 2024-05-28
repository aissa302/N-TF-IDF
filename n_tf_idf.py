import numpy as np
from collections import defaultdict
from math import log, sqrt

def compute_tf(category_word_counts, total_word_count_in_category):
    tf = {}
    for word, count in category_word_counts.items():
        tf[word] = count / total_word_count_in_category
    return tf

def compute_idf(categories_word_counts, total_docs):
    idf = {}
    total_word_occurrences = defaultdict(int)

    for category, word_counts in categories_word_counts.items():
        for word in word_counts:
            total_word_occurrences[word] += word_counts[word]

    for word, total_count in total_word_occurrences.items():
        doc_count_per_category = sum(1 for category in categories_word_counts if word in categories_word_counts[category])
        idf[word] = log((doc_count_per_category * total_docs) / total_count)

    return idf

def normalize_tf_idf(tf_idf, categories):
    norm_tf_idf = defaultdict(dict)
    
    for category in categories:
        norm_factor = sqrt(sum((tf_idf[category][word] ** 2 for word in tf_idf[category])))
        for word in tf_idf[category]:
            norm_tf_idf[category][word] = tf_idf[category][word] / norm_factor
    
    return norm_tf_idf

def compute_n_tf_idf(categories_word_counts, total_docs):
    tf = {}
    idf = compute_idf(categories_word_counts, total_docs)
    tf_idf = defaultdict(dict)
    
    for category, word_counts in categories_word_counts.items():
        total_word_count_in_category = sum(word_counts.values())
        tf[category] = compute_tf(word_counts, total_word_count_in_category)
        
        for word, tf_value in tf[category].items():
            tf_idf[category][word] = tf_value * idf[word]
    
    normalized_tf_idf = normalize_tf_idf(tf_idf, categories_word_counts.keys())
    
    return normalized_tf_idf

# Example usage:
categories_word_counts = {
    'category1': {'word1': 3, 'word2': 5, 'word3': 2},
    'category2': {'word1': 4, 'word2': 1, 'word3': 3},
    'category3': {'word1': 2, 'word2': 3, 'word3': 4},
}

total_docs = sum(len(word_counts) for word_counts in categories_word_counts.values())

n_tf_idf = compute_n_tf_idf(categories_word_counts, total_docs)
max_n_tf_idf_per_word = defaultdict(float)
for category, word_scores in n_tf_idf.items():
    print(f"Category: {category}")
    for word, score in word_scores.items():
        print(f"  Word: {word}, N-TF-IDF: {score}")

# Iterate over each category's N-TF-IDF values to find the maximum for each word
for category, word_scores in n_tf_idf.items():
    for word, score in word_scores.items():
        # Update the maximum N-TF-IDF value for each word
        max_n_tf_idf_per_word[word] = max(max_n_tf_idf_per_word[word], score)

# Print the maximum N-TF-IDF value for each word
print("Maximum N-TF-IDF values for each word:")
for word, max_score in max_n_tf_idf_per_word.items():
    print(f"Word: {word}, Max N-TF-IDF: {max_score}")