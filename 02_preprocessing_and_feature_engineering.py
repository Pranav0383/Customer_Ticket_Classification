# ==============================================
# Phase 3: Data Cleaning & Text Preprocessing
# Phase 4: Feature Engineering
# ==============================================
#
# CHANGES vs original:
# 1. Numeric features (has_attachment, text_length, word_count, urgency_count)
#    are now scaled with MaxAbsScaler before being combined with TF-IDF/OneHot.
#    Reason: raw text_length/word_count can range into the hundreds while
#    TF-IDF and OneHot columns are 0-1. Unscaled, regularized linear models
#    (Logistic Regression, LinearSVC) effectively ignore the numeric columns
#    or let them dominate unpredictably. MaxAbsScaler keeps everything
#    sparse-matrix-compatible (unlike StandardScaler, which densifies).
# 2. We save df['cleaned_message'] alongside the feature matrix as
#    "message_groups.pkl". The training scripts (03/04) use this to do a
#    GROUP-aware train/test split instead of a plain random split, so that
#    rows sharing identical message text (common with templated data) always
#    land entirely in train OR entirely in test -- never both. That's what
#    was silently causing the issue_type model's overfitting: identical text
#    leaking across the split let it "memorize" test answers.

import pandas as pd
import numpy as np
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent
nltk.download('stopwords', quiet=True)

# Load the cleaned dataset from Phase 1
df = pd.read_csv(PROJECT_DIR / "cleaned_ticket_data.csv")

print("Dataset Loaded:", df.shape)
print(df.head())

# ==============================================
# Phase 3: Data Cleaning & Text Preprocessing
# ==============================================

print("\n" + "="*50)
print("Phase 3: Data Cleaning & Text Preprocessing")
print("="*50)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df['region'] = df['region'].fillna('Unknown')

print("\nMissing values after cleaning:")
print(df.isnull().sum())

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned_message'] = df['initial_message'].apply(clean_text)

print("\nExample of Cleaned Text:")
print("Original:", df['initial_message'].iloc[0])
print("Cleaned :", df['cleaned_message'].iloc[0])

# IMPORTANT: drop exact-duplicate rows first. With templated data, the same
# cleaned_message + same categorical context can appear more than once;
# leaving true duplicates in doesn't add information and slightly inflates
# apparent performance either way. This is separate from the group-split
# fix below (which handles *text* reused across *different* rows).
before = len(df)
df = df.drop_duplicates(subset=['cleaned_message', 'customer_segment', 'channel',
                                 'product_area', 'issue_type', 'priority']).reset_index(drop=True)
print(f"\nDropped {before - len(df)} exact-duplicate rows ({len(df)} remaining)")

df['text_length'] = df['cleaned_message'].apply(len)
df['word_count'] = df['cleaned_message'].apply(lambda x: len(x.split()))

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

# 3. Numerical Features -- now scaled (see note at top of file)
X_num = df[['has_attachment', 'text_length', 'word_count', 'urgency_count']].values.astype(float)
num_scaler = MaxAbsScaler()
X_num_scaled = num_scaler.fit_transform(X_num)
print("Numerical features shape:", X_num_scaled.shape)

X_num_sparse = csr_matrix(X_num_scaled)

# 4. Combine All Features
X = hstack([X_text, X_cat, X_num_sparse]).tocsr()
print("\nFinal Combined Feature Matrix shape:", X.shape)

# 5. Targets
y_issue = df['issue_type']
y_priority = df['priority']

print("\nIssue Type classes:", y_issue.nunique())
print("Priority classes:", y_priority.nunique())

# 6. Save processed data for next phases
import joblib

joblib.dump(X, PROJECT_DIR / "X_features.pkl")
joblib.dump(y_issue, PROJECT_DIR / "y_issue.pkl")
joblib.dump(y_priority, PROJECT_DIR / "y_priority.pkl")
joblib.dump(tfidf, PROJECT_DIR / "tfidf_vectorizer.pkl")
joblib.dump(encoder, PROJECT_DIR / "onehot_encoder.pkl")
joblib.dump(num_scaler, PROJECT_DIR / "num_scaler.pkl")
# group key for leak-safe splitting in 03/04 -- same row order as X
joblib.dump(df['cleaned_message'].reset_index(drop=True), PROJECT_DIR / "message_groups.pkl")

print("\nAll processed files saved successfully!")
print("Phase 3 and Phase 4 Completed!")
