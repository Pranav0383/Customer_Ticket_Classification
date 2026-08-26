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
num_scaler = joblib.load(project_dir / "num_scaler.pkl")  # NEW: must match training-time scaling

print("All models loaded successfully!")

# ----------------------------------------------------------
# 2. Text cleaning function (must match your Phase 3)
# ----------------------------------------------------------
def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
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
    NOTE: `customer_tier` here is passed straight through to the
    'customer_segment' column the encoder expects -- keep the values as
    one of individual/small_business/enterprise/education/non_profit.
    """

    cleaned_message = clean_text(initial_message)

    tfidf_features = tfidf.transform([cleaned_message])

    cat_df = pd.DataFrame({
        'customer_segment': [str(customer_tier).lower()],
        'channel': [str(channel).lower()],
        'product_area': [str(product_area).lower()],
        'platform': [str(platform).lower()],
        'region': [str(region).upper()],
        'customer_sentiment': [str(customer_sentiment).lower()]
    })
    cat_features = onehot.transform(cat_df)

    raw_numeric = np.array([[
        has_attachment,
        len(cleaned_message),
        len(cleaned_message.split()),
        sum(1 for keyword in urgency_keywords if keyword in cleaned_message)
    ]], dtype=float)
    # scale with the SAME scaler fit during training -- this was missing
    # before and is required now that 02_preprocessing scales numeric features
    scaled_numeric = num_scaler.transform(raw_numeric)
    numerical_features = csr_matrix(scaled_numeric)

    X_new = hstack([tfidf_features, cat_features, numerical_features])

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
        customer_tier="enterprise",
        channel="email",
        product_area="billing"
    )

    print("\n--- Sample Prediction ---")
    print(result)
