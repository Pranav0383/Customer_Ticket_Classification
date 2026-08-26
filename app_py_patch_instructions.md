# app.py — 2 edits needed

## Edit 1: load the scaler (in `load_models()`)

Change:
```python
@st.cache_resource
def load_models():
    return (
        joblib.load(PROJECT_DIR / "best_issue_type_model.pkl"),
        joblib.load(PROJECT_DIR / "best_priority_model.pkl"),
        joblib.load(PROJECT_DIR / "tfidf_vectorizer.pkl"),
        joblib.load(PROJECT_DIR / "onehot_encoder.pkl"),
        joblib.load(PROJECT_DIR / "num_scaler.pkl"),
    )
```

To:
```python
@st.cache_resource
def load_models():
    return (
        joblib.load(PROJECT_DIR / "best_issue_type_model.pkl"),
        joblib.load(PROJECT_DIR / "best_priority_model.pkl"),
        joblib.load(PROJECT_DIR / "tfidf_vectorizer.pkl"),
        joblib.load(PROJECT_DIR / "onehot_encoder.pkl"),
        joblib.load(PROJECT_DIR / "num_scaler.pkl"),
    )
```

## Edit 2: apply it in `predict_ticket()`

Change:
```python
def predict_ticket(initial_message, customer_tier, channel, product_area,
                   platform="web", region="NA", customer_sentiment="neutral",
                   has_attachment=0):
    issue_model, priority_model, tfidf, onehot = load_models()
    cleaned_message = clean_text(initial_message)
    category_values = pd.DataFrame({
        "customer_segment": [str(customer_tier).lower()],
        "channel": [str(channel).lower()],
        "product_area": [str(product_area).lower()],
        "platform": [str(platform).lower()],
        "region": [str(region).upper()],
        "customer_sentiment": [str(customer_sentiment).lower()],
    })
    numeric_values = csr_matrix([[
        int(has_attachment),
        len(cleaned_message),
        len(cleaned_message.split()),
        sum(keyword in cleaned_message for keyword in URGENCY_KEYWORDS),
    ]])
    features = hstack([
        tfidf.transform([cleaned_message]),
        onehot.transform(category_values),
        numeric_values,
    ])
```

To:
```python
def predict_ticket(initial_message, customer_tier, channel, product_area,
                   platform="web", region="NA", customer_sentiment="neutral",
                   has_attachment=0):
    issue_model, priority_model, tfidf, onehot, num_scaler = load_models()
    cleaned_message = clean_text(initial_message)
    category_values = pd.DataFrame({
        "customer_segment": [str(customer_tier).lower()],
        "channel": [str(channel).lower()],
        "product_area": [str(product_area).lower()],
        "platform": [str(platform).lower()],
        "region": [str(region).upper()],
        "customer_sentiment": [str(customer_sentiment).lower()],
    })
    raw_numeric = [[
        int(has_attachment),
        len(cleaned_message),
        len(cleaned_message.split()),
        sum(keyword in cleaned_message for keyword in URGENCY_KEYWORDS),
    ]]
    numeric_values = csr_matrix(num_scaler.transform(raw_numeric))
    features = hstack([
        tfidf.transform([cleaned_message]),
        onehot.transform(category_values),
        numeric_values,
    ])
```

Nothing else in app.py needs to change — the selectbox values already match
the new dataset's category values exactly (customer_segment options,
channel options, product_area options all line up).
