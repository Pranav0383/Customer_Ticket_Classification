import joblib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ----------------------------------------------------------
# Load models (files are in the same folder)
# ----------------------------------------------------------
project_dir = Path(__file__).resolve().parent
issue_model = joblib.load(project_dir / "best_issue_type_model.pkl")
priority_model = joblib.load(project_dir / "best_priority_model.pkl")
tfidf = joblib.load(project_dir / "tfidf_vectorizer.pkl")
onehot = joblib.load(project_dir / "onehot_encoder.pkl")

print("All models loaded successfully!")

# ----------------------------------------------------------
# 2. Text cleaning function (must match your Phase 3)
# ----------------------------------------------------------
def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)      # remove special characters & numbers
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


urgency_keywords = [
    'urgent', 'asap', 'immediately', 'critical', 'emergency',
    'not working', 'down', 'error', 'failed', 'crash', 'blocked'
]

# ----------------------------------------------------------
# 3. Main Prediction Function
# ----------------------------------------------------------
def predict_ticket(
    initial_message,
    customer_tier,
    channel,
    product_area,
    platform='web',
    region='NA',
    customer_sentiment='neutral',
    has_attachment=0
):
    """
    Predict issue_type and priority for a new support ticket.
    """

    # Clean the message
    cleaned_message = clean_text(initial_message)

    # TF-IDF transform
    tfidf_features = tfidf.transform([cleaned_message])

    # Prepare categorical features for OneHotEncoder
    # IMPORTANT: Column order must match what you used during training
    cat_df = pd.DataFrame({
        'customer_segment': [str(customer_tier).lower()],
        'channel': [str(channel).lower()],
        'product_area': [str(product_area).lower()],
        'platform': [str(platform).lower()],
        'region': [str(region).upper()],
        'customer_sentiment': [str(customer_sentiment).lower()]
    })

    # One-Hot Encode
    cat_features = onehot.transform(cat_df)

    # Add numerical features in the same order as the training pipeline.
    numerical_features = csr_matrix([[
        has_attachment,
        len(cleaned_message),
        len(cleaned_message.split()),
        sum(1 for keyword in urgency_keywords if keyword in cleaned_message)
    ]])

    # Combine features in the same order as training.
    X_new = hstack([tfidf_features, cat_features, numerical_features])

    # Make predictions
    issue_pred = issue_model.predict(X_new)[0]
    priority_pred = priority_model.predict(X_new)[0]

    return {
        "Predicted Issue Type": issue_pred,
        "Predicted Priority": priority_pred
    }

# ----------------------------------------------------------
# 4. Test the function
# ----------------------------------------------------------
if __name__ == "__main__":
    result = predict_ticket(
        initial_message="My payment failed and I cannot login to the dashboard. This is urgent!",
        customer_tier="Premium",
        channel="Email",
        product_area="Billing"
    )

    print("\n--- Sample Prediction ---")
    print(result)