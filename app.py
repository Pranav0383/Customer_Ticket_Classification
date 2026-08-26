import math
import re
from pathlib import Path

import joblib
import importlib
import nltk
import pandas as pd
import streamlit as st  # type: ignore[import-not-found]
scipy_sparse = importlib.import_module("scipy.sparse")
csr_matrix = scipy_sparse.csr_matrix
hstack = scipy_sparse.hstack

nltk.download("stopwords", quiet=True)
stopwords = importlib.import_module("nltk.corpus").stopwords


st.set_page_config(page_title="Ticket Lens", page_icon="TL", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
URGENCY_KEYWORDS = [
    "urgent", "asap", "immediately", "critical", "emergency",
    "not working", "down", "error", "failed", "crash", "blocked",
]


@st.cache_resource
def load_models():
    return (
        joblib.load(PROJECT_DIR / "best_issue_type_model.pkl"),
        joblib.load(PROJECT_DIR / "best_priority_model.pkl"),
        joblib.load(PROJECT_DIR / "tfidf_vectorizer.pkl"),
        joblib.load(PROJECT_DIR / "onehot_encoder.pkl"),
        joblib.load(PROJECT_DIR / "num_scaler.pkl"),
    )

@st.cache_resource
def load_stop_words():
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except LookupError:
        return set()


def clean_text(text):
    if text is None or pd.isna(text):
        return ""
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in load_stop_words())


def model_signal(model, features):
    if not hasattr(model, "decision_function"):
        return None
    scores = model.decision_function(features)[0]
    scores = scores if hasattr(scores, "__len__") else [scores]
    peak = max(scores)
    weights = [math.exp(float(score) - peak) for score in scores]
    return max(weights) / sum(weights) if weights else None


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
    issue_type = issue_model.predict(features)[0]
    priority = priority_model.predict(features)[0]
    issue_signal = model_signal(issue_model, features)
    priority_signal = model_signal(priority_model, features)
    return issue_type, priority, issue_signal, priority_signal

def title_case(value):
    return str(value).replace("_", " ").title()


def ticket_insights(message, sentiment, attachment):
    cleaned = clean_text(message)
    matched = [keyword for keyword in URGENCY_KEYWORDS if keyword in cleaned]
    return {
        "words": len(cleaned.split()),
        "urgency_terms": len(matched),
        "sentiment": title_case(sentiment),
        "attachment": "Included" if attachment else "None",
        "matched": matched,
    }


def priority_guidance(priority):
    return {
        "urgent": ("Immediate response", "Route to the on-call queue and acknowledge the customer now."),
        "high": ("Respond today", "Assign an owner and begin investigation within the current shift."),
        "medium": ("Standard queue", "Add context, confirm scope, and work through the normal SLA."),
        "low": ("Planned follow-up", "Bundle with routine work and respond when the queue allows."),
    }.get(priority.lower(), ("Review required", "Confirm the classification before routing."))


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');

:root {
  --ink:#152b35; --muted:#5c7379; --line:#dbe6e3; --teal:#0f7a72; --teal-dark:#0b514f;
  --coral:#ef765d; --panel:#ffffff; --panel-soft:#fbfcfa; --shadow:0 10px 30px -14px rgba(21,43,53,.18);
}

/* ---------- base / resets so nothing inherits an invisible theme color ---------- */
html, body, [class^="css"], .stApp, .stApp * { color: var(--ink); }
.stApp {
  background: radial-gradient(circle at 92% -4%, #ffe6c9 0, #ffe6c900 32rem),
              radial-gradient(circle at -6% 110%, #d7f0e6 0, #d7f0e600 30rem),
              linear-gradient(160deg,#f7faf8 0%,#eef6f2 55%,#f6f1e9 100%);
}
.block-container { max-width:1200px; padding:2rem 2rem 3.5rem; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer[class]{ visibility:hidden; }

/* ---------- masthead ---------- */
.masthead {
  display:flex; justify-content:space-between; align-items:flex-end; gap:2rem;
  padding:1.75rem 2rem; margin-bottom:1.75rem; border-radius:18px;
  background:linear-gradient(120deg,#ffffff, #f4faf7 70%); border:1px solid var(--line); box-shadow:var(--shadow);
}
.kicker { color:var(--teal); font:600 .72rem 'DM Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }
.masthead h1 { margin:.5rem 0 .4rem; color:var(--ink) !important; font:800 2.6rem/1.05 Manrope,sans-serif; letter-spacing:-.01em; }
.masthead p { margin:0; color:var(--muted) !important; font:400 1.02rem/1.55 Manrope,sans-serif; }
.status-pill {
  padding:.6rem .95rem; border:1px solid #b6ddd0; border-radius:999px; background:#eafaf3;
  color:var(--teal) !important; font:700 .8rem Manrope,sans-serif; white-space:nowrap; box-shadow:0 2px 8px -4px rgba(15,122,114,.35);
}

/* ---------- section titles ---------- */
.section-title { display:flex; align-items:center; gap:.6rem; margin:1.75rem 0 .3rem; color:var(--ink) !important; font:800 1.12rem Manrope,sans-serif; }
.section-title span {
  display:inline-flex; align-items:center; justify-content:center; width:1.7rem; height:1.7rem;
  color:#fff !important; background:var(--coral); border-radius:8px; font:700 .72rem 'DM Mono',monospace;
}
.subcopy { margin:0 0 1rem; color:var(--muted) !important; font:.88rem/1.55 Manrope,sans-serif; }

/* ---------- form "card" wrapper ---------- */
div[data-testid="stForm"] {
  background:var(--panel); border:1px solid var(--line); border-radius:18px;
  padding:1.75rem 1.85rem 1.5rem; box-shadow:var(--shadow);
}

/* ---------- text area ---------- */
div[data-testid="stTextArea"] textarea {
  background:var(--panel-soft) !important; border:1.5px solid #c6d9d4 !important; border-radius:12px;
  color:var(--ink) !important; font:400 1rem/1.6 Manrope,sans-serif; padding:1rem;
  transition:border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stTextArea"] textarea::placeholder { color:#96a9ad !important; opacity:1 !important; }
div[data-testid="stTextArea"] textarea:focus { border-color:var(--teal) !important; box-shadow:0 0 0 4px #c9ece3; }

/* ---------- labels for all widgets (select / checkbox / text) ---------- */
div[data-testid="stSelectbox"] label p,
div[data-testid="stCheckbox"] label p,
div[data-testid="stTextArea"] label p,
.stSelectbox label, .stCheckbox label {
  color:var(--ink) !important; font-weight:700 !important; font-family:Manrope,sans-serif;
}

/* ---------- selectbox control ---------- */
div[data-testid="stSelectbox"] > div > div {
  background:#16323a !important; border-radius:10px !important; border:1px solid #0f2830 !important;
}
div[data-testid="stSelectbox"] * { color:#fdfefe !important; }
div[data-testid="stSelectbox"] svg { fill:#fdfefe !important; }

/* ---------- checkbox text ---------- */
div[data-testid="stCheckbox"] label span p { color:var(--ink) !important; }

/* ---------- submit button ---------- */
button[kind="primaryFormSubmit"] {
  min-height:3.2rem; border:0; border-radius:10px;
  background:linear-gradient(120deg,var(--teal),#0c5f5a); color:#fff !important;
  font:800 .95rem Manrope,sans-serif; letter-spacing:.01em; transition:transform .12s ease, box-shadow .12s ease;
  box-shadow:0 10px 22px -10px rgba(15,122,114,.55);
}
button[kind="primaryFormSubmit"] p { color:#fff !important; }
button[kind="primaryFormSubmit"]:hover { transform:translateY(-1px); box-shadow:0 14px 26px -10px rgba(15,122,114,.65); }
button[kind="primaryFormSubmit"]:active { transform:translateY(0); }

/* ---------- alerts (warning / success / error) ---------- */
div[data-testid="stAlert"] { border-radius:12px !important; border:1px solid transparent !important; box-shadow:var(--shadow); }
div[data-testid="stAlert"] p, div[data-testid="stAlert"] div, div[data-testid="stAlert"] span { color:var(--ink) !important; font-weight:500; }
div[data-baseweb="notification"] { background:#fff8e6 !important; }

/* ---------- result panels ---------- */
.result-panel { height:100%; padding:1.5rem; border:1px solid var(--line); border-radius:16px; background:var(--panel); box-shadow:var(--shadow); }
.result-label { color:var(--muted) !important; font:600 .72rem 'DM Mono',monospace; letter-spacing:.07em; text-transform:uppercase; }
.result-value { margin:.5rem 0 1.1rem; color:var(--ink) !important; font:800 1.6rem Manrope,sans-serif; }
.priority-value { color:#c8412e !important; }
.insight { padding:1rem 1.1rem; border-left:4px solid var(--coral); background:#fff6f0; border-radius:0 10px 10px 0; color:var(--ink) !important; font:.9rem/1.55 Manrope,sans-serif; }
.insight strong { color:var(--ink) !important; }
.metric-strip { margin-top:1.1rem; padding:1rem 0 .2rem; border-top:1px solid var(--line); }
.metric-number { color:var(--teal) !important; font:800 1.25rem Manrope,sans-serif; }
.metric-name { color:var(--muted) !important; font:.7rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.04em; }

/* ---------- streamlit st.metric widgets ---------- */
div[data-testid="stMetric"] { background:var(--panel-soft); border:1px solid var(--line); border-radius:12px; padding:.7rem .9rem; }
div[data-testid="stMetricLabel"] p { color:var(--muted) !important; font:600 .72rem 'DM Mono',monospace; text-transform:uppercase; }
div[data-testid="stMetricValue"] { color:var(--ink) !important; font:800 1.4rem Manrope,sans-serif; }

/* ---------- caption ---------- */
div[data-testid="stCaptionContainer"] p { color:var(--muted) !important; font:.82rem Manrope,sans-serif; }

/* ---------- history strip ---------- */
.history-item {
  padding:.9rem 1rem; border:1px solid var(--line); border-radius:12px; background:var(--panel);
  color:var(--ink) !important; font:.86rem Manrope,sans-serif; box-shadow:var(--shadow);
}
.history-item strong { color:var(--ink) !important; }
.history-item small { color:var(--muted) !important; }

.footer { margin-top:2.5rem; text-align:center; color:#8aa09d !important; font:.72rem 'DM Mono',monospace; letter-spacing:.06em; }

@media (max-width: 700px) {
  .masthead { display:block; }
  .status-pill { display:inline-block; margin-top:1rem; }
  .masthead h1 { font-size:2.05rem; }
  .block-container { padding:1.35rem 1rem 2rem; }
  div[data-testid="stForm"] { padding:1.25rem 1.1rem; }
}
</style>
""", unsafe_allow_html=True)


try:
    load_models()
except Exception as error:
    st.error("The classifier models could not be loaded.")
    st.code(f"{type(error).__name__}: {error}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<div class="masthead">
  <div><div class="kicker">Support operations / Ticket Lens</div>
  <h1>Make every ticket actionable.</h1>
  <p>Classify issue type, urgency, and next response in one focused pass.</p></div>
  <div class="status-pill">● Models online</div>
</div>
""", unsafe_allow_html=True)

with st.form("ticket_form"):
    st.markdown('<div class="section-title"><span>01</span>Customer message</div>', unsafe_allow_html=True)
    st.markdown('<div class="subcopy">The message is cleaned and scored for intent and urgency signals.</div>', unsafe_allow_html=True)
    initial_message = st.text_area("Customer message", height=145, label_visibility="collapsed", placeholder="Example: My payment failed and I cannot access the dashboard. This is urgent!")

    st.markdown('<div class="section-title"><span>02</span>Ticket context</div>', unsafe_allow_html=True)
    st.markdown('<div class="subcopy">Context helps separate similar requests and improves routing.</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_tier = st.selectbox("Customer segment", ["individual", "small_business", "enterprise", "education", "non_profit"])
    with col2:
        channel = st.selectbox("Channel", ["email", "chat", "phone_transcript", "in_app", "web_form"])
    with col3:
        product_area = st.selectbox("Product area", ["billing", "api_integration", "analytics_dashboard", "login_auth", "mobile_app", "notifications", "data_export"])
    col4, col5, col6 = st.columns(3)
    with col4:
        platform = st.selectbox("Platform", ["web", "ios", "android", "desktop_app", "api_client"])
    with col5:
        region = st.selectbox("Region", ["NA", "EU", "APAC", "LATAM", "MEA"])
    with col6:
        customer_sentiment = st.selectbox("Customer sentiment", ["neutral", "positive", "negative", "very_positive", "very_negative"])
    has_attachment = st.checkbox("This ticket includes an attachment")
    submitted = st.form_submit_button("Analyze ticket", type="primary", use_container_width=True)

if submitted:
    if not initial_message.strip():
        st.warning("Add the customer message before analyzing the ticket.")
    else:
        try:
            with st.spinner("Reading ticket signals..."):
                issue_type, priority, issue_signal, priority_signal = predict_ticket(initial_message, customer_tier, channel, product_area, platform, region, customer_sentiment, int(has_attachment))
            insights = ticket_insights(initial_message, customer_sentiment, has_attachment)
            action, guidance = priority_guidance(priority)
            issue_confidence = f"{issue_signal:.0%}" if issue_signal is not None else "Unavailable"
            priority_confidence = f"{priority_signal:.0%}" if priority_signal is not None else "Unavailable"
            st.session_state.history.insert(0, {"issue": title_case(issue_type), "priority": priority.title(), "area": title_case(product_area)})
            st.session_state.history = st.session_state.history[:4]
            st.markdown('<div class="section-title"><span>03</span>Routing recommendation</div>', unsafe_allow_html=True)
            left, right = st.columns([1.15, .85])
            with left:
                st.markdown(f'<div class="result-panel"><div class="result-label">Predicted issue type</div><div class="result-value">{title_case(issue_type)}</div><div class="result-label">Recommended handling</div><div class="result-value priority-value">{priority.title()}</div><div class="insight"><strong>{action}</strong><br>{guidance}</div><div class="metric-strip"><div class="metric-number">{issue_confidence} / {priority_confidence}</div><div class="metric-name">Issue / priority model signal</div></div></div>', unsafe_allow_html=True)
            with right:
                st.markdown('<div class="result-panel"><div class="result-label">Ticket signals</div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Message words", insights["words"])
                m2.metric("Urgency terms", insights["urgency_terms"])
                st.markdown(f'<div class="metric-strip"><div class="metric-number">{insights["sentiment"]}</div><div class="metric-name">Customer sentiment</div><br><div class="metric-number">{insights["attachment"]}</div><div class="metric-name">Attachment status</div></div></div>', unsafe_allow_html=True)
            if insights["matched"]:
                st.caption("Signals found: " + ", ".join(insights["matched"]))
            st.success("Ticket analyzed and ready to route.")
        except Exception as error:
            st.error("This ticket could not be analyzed. Check the context values and try again.")
            st.exception(error)

if st.session_state.history:
    st.markdown('<div class="section-title"><span>04</span>Recent analyses</div>', unsafe_allow_html=True)
    history_cols = st.columns(len(st.session_state.history))
    for column, item in zip(history_cols, st.session_state.history):
        with column:
            st.markdown(f'<div class="history-item"><strong>{item["issue"]}</strong><br><small>{item["area"]} · {item["priority"]}</small></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">TICKET LENS  /  CLASSIFICATION WORKBENCH  /  LOCAL MODEL INFERENCE</div>', unsafe_allow_html=True)