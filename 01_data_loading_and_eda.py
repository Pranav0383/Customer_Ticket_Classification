# ==============================================
# Phase 1: Project Setup & Understanding
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')



# -----------------------------
# 1. Load the Dataset
# -----------------------------
# Replace 'your_dataset.csv' with your actual file name
df = pd.read_csv("synthetic_it_support_tickets.csv")

print("Dataset Loaded Successfully!")
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

# -----------------------------
# 2. Keep only Required Columns
# -----------------------------
required_columns = [
    'initial_message',
    'customer_segment',
    'channel',
    'product_area',
    'platform',
    'region',
    'has_attachment',
    'customer_sentiment',
    'issue_type',
    'priority'
]

df = df[required_columns]

print("\nDataset after selecting required columns:")
print(df.shape)
print(df.head())

# Optional: Save cleaned version
df.to_csv("cleaned_ticket_data.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_ticket_data.csv'")

# ==============================================
# Phase 2: Exploratory Data Analysis (EDA)
# ==============================================

print("\n" + "="*50)
print("Starting Exploratory Data Analysis")
print("="*50)

# 1. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Basic Statistics
print("\nDataset Description:")
print(df.describe(include='all'))

# 3. Target Distribution - Issue Type
print("\nIssue Type Distribution:")
print(df['issue_type'].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(y='issue_type', data=df, order=df['issue_type'].value_counts().index)
plt.title("Distribution of Issue Type")
plt.xlabel("Count")
plt.ylabel("Issue Type")
plt.tight_layout()
plt.show()

# 4. Target Distribution - Priority
print("\nPriority Distribution:")
print(df['priority'].value_counts())

plt.figure(figsize=(7,4))
sns.countplot(x='priority', data=df, order=df['priority'].value_counts().index)
plt.title("Distribution of Priority")
plt.xlabel("Priority")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. Analyze Text Length
df['text_length'] = df['initial_message'].astype(str).apply(len)
df['word_count'] = df['initial_message'].astype(str).apply(lambda x: len(x.split()))

print("\nText Length Statistics:")
print(df['text_length'].describe())

print("\nWord Count Statistics:")
print(df['word_count'].describe())

plt.figure(figsize=(10,4))
sns.histplot(data=df, x='text_length', bins=50, kde=True)
plt.title("Distribution of Text Length")
plt.xlabel("Text Length")
plt.show()

# 6. Categorical Features Analysis
categorical_cols = ['customer_tier', 'channel', 'product_area', 'platform', 'region', 'customer_sentiment']

for col in categorical_cols:
    print(f"\n{col} Distribution:")
    print(df[col].value_counts())
    
    plt.figure(figsize=(8,4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

# 7. Check relationship between Priority and some features (Optional)
plt.figure(figsize=(8,4))
sns.countplot(x='priority', hue='customer_tier', data=df)
plt.title("Priority vs Customer Tier")
plt.show()

print("\n" + "="*50)
print("Phase 1 and Phase 2 Completed Successfully!")
print("="*50)