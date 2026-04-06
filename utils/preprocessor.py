# =============================================================
# utils/preprocessor.py
# PURPOSE: Handles all text cleaning and feature extraction
# for the spam detection model.
# =============================================================

import re                          # Regular expressions for text cleaning
import string                      # String constants (punctuation, etc.)
import pandas as pd                # Data manipulation library
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to numbers


def clean_text(text):
    """
    Cleans raw email text by removing noise.

    Steps:
      1. Convert to lowercase (so 'FREE' and 'free' are treated the same)
      2. Remove URLs (http links)
      3. Remove email addresses
      4. Remove punctuation
      5. Remove extra whitespace

    Args:
        text (str): Raw email text

    Returns:
        str: Cleaned text string
    """
    # Step 1: Lowercase everything
    text = text.lower()

    # Step 2: Remove URLs like http://example.com
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove email addresses like user@example.com
    text = re.sub(r'\S+@\S+', '', text)

    # Step 4: Remove punctuation (commas, exclamation marks, etc.)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 5: Strip extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_and_preprocess(filepath):
    """
    Loads the dataset from a CSV file and cleans the text column.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: DataFrame with original + cleaned text
    """
    # Load CSV into a DataFrame
    df = pd.read_csv(filepath)

    # Apply the clean_text function to every row in the 'text' column
    df['clean_text'] = df['text'].apply(clean_text)

    # Convert labels: 'spam' -> 1, 'ham' -> 0
    # This is needed because ML models work with numbers, not strings
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

    print(f"✅ Dataset loaded: {len(df)} emails ({df['label'].value_counts()['spam']} spam, {df['label'].value_counts()['ham']} ham)")

    return df


def get_vectorizer():
    """
    Returns a TF-IDF Vectorizer.

    TF-IDF (Term Frequency-Inverse Document Frequency) converts text
    into a matrix of numbers. Words that appear often in spam but rarely
    in normal emails get higher scores — helping the model learn patterns.

    Settings:
        - max_features: Only keep the top 3000 most useful words
        - ngram_range: Use both single words AND pairs of words
                       e.g., "free money" is more spammy than "free" alone
        - stop_words: Ignore common English words like 'the', 'is', 'at'
    """
    vectorizer = TfidfVectorizer(
        max_features=3000,       # Limit vocabulary size
        ngram_range=(1, 2),      # Use unigrams and bigrams
        stop_words='english'     # Remove common filler words
    )
    return vectorizer
