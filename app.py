# =============================================================
# app.py
# PURPOSE: Web interface for the spam detector.
# Loads the saved model and serves a browser-based UI.
#
# HOW TO RUN:
#   python app.py
# Then open your browser at: http://127.0.0.1:5000
# =============================================================

import pickle
from flask import Flask, render_template, request, jsonify
from utils.preprocessor import clean_text

# Initialize the Flask web application
app = Flask(__name__)


# ─── Load the saved model and vectorizer ─────────────────────
# These files are created by running train_model.py first
try:
    with open('models/spam_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    print("✅ Model and vectorizer loaded successfully!")

except FileNotFoundError:
    print("❌ ERROR: Model files not found.")
    print("   Please run: python train_model.py first!")
    model = None
    vectorizer = None


def predict_spam(email_text):
    """
    Takes raw email text and returns a prediction.

    Args:
        email_text (str): The raw email content

    Returns:
        dict: Contains label, confidence score, and cleaned text
    """
    if model is None or vectorizer is None:
        return {"error": "Model not loaded. Run train_model.py first."}

    # Step 1: Clean the input text (same preprocessing as training)
    cleaned = clean_text(email_text)

    # Step 2: Vectorize using the SAME vectorizer from training
    vectorized = vectorizer.transform([cleaned])

    # Step 3: Get prediction (0 = ham, 1 = spam)
    prediction = model.predict(vectorized)[0]

    # Step 4: Get probability scores for both classes
    # predict_proba returns [P(ham), P(spam)]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "label": "SPAM" if prediction == 1 else "NOT SPAM",
        "is_spam": bool(prediction == 1),
        "confidence": confidence,
        "spam_probability": round(float(probabilities[1]) * 100, 2),
        "ham_probability": round(float(probabilities[0]) * 100, 2),
    }


# ─── Routes ───────────────────────────────────────────────────

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint: receives email text, returns prediction as JSON.
    Called by the browser's JavaScript when user clicks 'Analyze'.
    """
    data = request.get_json()
    email_text = data.get('email_text', '').strip()

    if not email_text:
        return jsonify({"error": "Please enter some email text."}), 400

    result = predict_spam(email_text)
    return jsonify(result)


# ─── Run the app ──────────────────────────────────────────────
if __name__ == '__main__':
    # debug=True: Automatically restarts when code changes (useful during development)
    # Remove debug=True in production
    app.run(debug=True, port=5000)
