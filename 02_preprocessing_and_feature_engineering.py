# ==============================================
# Phase 3: Data Cleaning & Text Preprocessing
# Phase 4: Feature Engineering
# ==============================================

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset from Phase 1
df = pd.read_csv("cleaned_ticket_data.csv")

print("Dataset Loaded:", df.shape)
print(df.head())

# ==============================================
# Phase 3: Data Cleaning & Text Preprocessing
# ==============================================

print("\n" + "="*50)
print("Phase 3: Data Cleaning & Text Preprocessing")
print("="*50)

# 1. Handle Missing Values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing region with 'Unknown'
df['region'] = df['region'].fillna('Unknown')

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 2. Text Cleaning Function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()                          # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)            # Remove special characters & numbers
    text = re.sub(r'\s+', ' ', text).strip()           # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Apply text cleaning
df['cleaned_message'] = df['initial_message'].apply(clean_text)

print("\nExample of Cleaned Text:")
print("Original:", df['initial_message'].iloc[0])
print("Cleaned :", df['cleaned_message'].iloc[0])

# 3. Create Extra Features
df['text_length'] = df['cleaned_message'].apply(len)
df['word_count'] = df['cleaned_message'].apply(lambda x: len(x.split()))

# Urgency keywords
urgency_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency', 
                    'not working', 'down', 'error', 'failed', 'crash', 'blocked']

def count_urgency(text):
    return sum(1 for word in urgency_keywords if word in text)

df['urgency_count'] = df['cleaned_message'].apply(count_urgency)

print("\nNew features created: text_length, word_count, urgency_count")
print(df[['cleaned_message', 'text_length', 'word_count', 'urgency_count']].head())

# ==============================================
# Phase 4: Feature Engineering
# ==============================================

print("\n" + "="*50)
print("Phase 4: Feature Engineering")
print("="*50)

# 1. TF-IDF on cleaned text
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

X_text = tfidf.fit_transform(df['cleaned_message'])
print("TF-IDF shape:", X_text.shape)

# 2. Encode Categorical Features
categorical_cols = ['customer_segment', 'channel', 'product_area', 
                    'platform', 'region', 'customer_sentiment']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_cat = encoder.fit_transform(df[categorical_cols])
print("Categorical features shape:", X_cat.shape)

# 3. Numerical Features
X_num = df[['has_attachment', 'text_length', 'word_count', 'urgency_count']].values
print("Numerical features shape:", X_num.shape)

# 4. Combine All Features
from scipy.sparse import csr_matrix
X_num_sparse = csr_matrix(X_num)

X = hstack([X_text, X_cat, X_num_sparse])
print("\nFinal Combined Feature Matrix shape:", X.shape)

# 5. Targets
y_issue = df['issue_type']
y_priority = df['priority']

print("\nIssue Type classes:", y_issue.nunique())
print("Priority classes:", y_priority.nunique())

# 6. Save processed data for next phases
import joblib

joblib.dump(X, "X_features.pkl")
joblib.dump(y_issue, "y_issue.pkl")
joblib.dump(y_priority, "y_priority.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(encoder, "onehot_encoder.pkl")

print("\nAll processed files saved successfully!")
print("Phase 3 and Phase 4 Completed!")