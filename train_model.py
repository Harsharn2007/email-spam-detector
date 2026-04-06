# =============================================================
# train_model.py
# PURPOSE: Train the spam detection model and save it to disk.
# Run this ONCE before using the app.
#
# HOW TO RUN:
#   python train_model.py
# =============================================================

import os
import pickle                          # For saving/loading Python objects
import pandas as pd
from sklearn.model_selection import train_test_split  # Split data into train/test sets
from sklearn.naive_bayes import MultinomialNB         # Our main ML model
from sklearn.linear_model import LogisticRegression  # Alternative model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Import our custom preprocessing utilities
from utils.preprocessor import load_and_preprocess, get_vectorizer


# ─── STEP 1: Load and clean the dataset ──────────────────────
print("\n📂 Loading dataset...")
df = load_and_preprocess('data/emails.csv')


# ─── STEP 2: Split into Training and Testing sets ────────────
# We train on 80% of the data and test on the remaining 20%
# random_state=42 ensures reproducibility (same split every run)
X = df['clean_text']       # Features: the cleaned email text
y = df['label_num']        # Labels: 0 (ham) or 1 (spam)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% for testing
    random_state=42,
    stratify=y              # Keep same spam/ham ratio in both sets
)

print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# ─── STEP 3: Vectorize the text (convert words to numbers) ───
print("\n🔢 Vectorizing text with TF-IDF...")
vectorizer = get_vectorizer()

# fit_transform: Learn vocabulary from training data AND transform it
X_train_vec = vectorizer.fit_transform(X_train)

# transform only: Use the SAME vocabulary learned above on test data
X_test_vec = vectorizer.transform(X_test)


# ─── STEP 4: Train the model ─────────────────────────────────
# We use Multinomial Naive Bayes — a classic, fast algorithm for text
# classification. It works by calculating probabilities of each word
# appearing in spam vs. ham emails.
print("\n🤖 Training Naive Bayes model...")
model = MultinomialNB(alpha=0.1)   # alpha: smoothing parameter
model.fit(X_train_vec, y_train)


# ─── STEP 5: Evaluate model performance ──────────────────────
print("\n📈 Evaluating model...")
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham (Not Spam)', 'Spam']))

print("🔲 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives  (Ham correctly identified):  {cm[0][0]}")
print(f"   False Positives (Ham wrongly called Spam):   {cm[0][1]}")
print(f"   False Negatives (Spam missed):               {cm[1][0]}")
print(f"   True Positives  (Spam correctly caught):     {cm[1][1]}")


# ─── STEP 6: Save model and vectorizer to disk ───────────────
# We save the trained model so we don't have to retrain every time
os.makedirs('models', exist_ok=True)

with open('models/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n💾 Model saved to models/spam_model.pkl")
print("💾 Vectorizer saved to models/vectorizer.pkl")
print("\n🎉 Training complete! You can now run: python app.py\n")
