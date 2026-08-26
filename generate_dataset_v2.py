"""
Synthetic support ticket generator - v2, schema-matched to the existing
Ticket Lens pipeline (01-05 scripts + app.py).

Fixes vs. the original 'synthetic_it_support_tickets.csv':
1. Thousands of unique message texts (templated + slot-filled), not ~96
   repeated strings -- removes the issue_type memorization/leakage problem.
2. `priority` is computed from a transparent scoring rule tied to
   issue severity + customer_segment + channel + has_attachment + urgency
   language in the text (+ noise) -- gives the priority model real signal
   instead of pure randomness.
3. Column names and category values match EXACTLY what 01-05/app.py expect,
   so you can drop this file in as a straight replacement with no pipeline
   rewrites: initial_message, customer_segment, channel, product_area,
   platform, region, has_attachment, customer_sentiment, issue_type, priority.
"""

import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------
# Category values -- must match your app.py selectboxes exactly
# ----------------------------------------------------------------------
PRODUCT_AREAS = ["billing", "api_integration", "analytics_dashboard",
                  "login_auth", "mobile_app", "notifications", "data_export"]

ISSUE_TYPES = ["account_access", "billing_problem", "bug", "feature_request",
               "how_to", "other", "performance", "security_concern"]

CUSTOMER_SEGMENTS = ["individual", "small_business", "enterprise", "education", "non_profit"]
CHANNELS = ["email", "chat", "phone_transcript", "in_app", "web_form"]
PLATFORMS = ["web", "ios", "android", "desktop_app", "api_client"]
REGIONS = ["NA", "EU", "APAC", "LATAM", "MEA"]

ISSUE_SEVERITY = {
    "security_concern": 8, "account_access": 7, "bug": 6, "performance": 5,
    "billing_problem": 4, "other": 3, "how_to": 1, "feature_request": 1,
}
SEGMENT_WEIGHT = {"enterprise": 4, "small_business": 2, "education": 1, "non_profit": 1, "individual": 0}
CHANNEL_WEIGHT = {"phone_transcript": 3, "chat": 2, "in_app": 1, "web_form": 1, "email": 0}

# ----------------------------------------------------------------------
# Slot vocab
# ----------------------------------------------------------------------
PRODUCTS_HUMAN = {
    "billing": ["my billing", "my invoice", "my subscription", "my payment method"],
    "api_integration": ["the API", "the REST API", "the webhook integration", "the API v2 endpoint"],
    "analytics_dashboard": ["the dashboard", "the analytics dashboard", "the reporting view", "the charts page"],
    "login_auth": ["the login page", "the sign-in flow", "two-factor authentication", "SSO login"],
    "mobile_app": ["the mobile app", "the iOS app", "the Android app", "your app"],
    "notifications": ["email notifications", "push notifications", "alert settings", "notification preferences"],
    "data_export": ["the data export", "the CSV export", "the export tool", "bulk export"],
}
ERROR_CODES = ["500", "403", "timeout", "null pointer", "502 Bad Gateway", "connection refused", "429 rate limit"]
TIME_PHRASES = ["this morning", "yesterday", "for the past 3 days", "since the last update",
                "right now", "for a week now", "since Monday"]
AMOUNTS = ["$19.99", "$49", "$120", "$9.99", "$250", "$15.50"]
PLAN_NAMES = ["Basic", "Pro", "Team", "Enterprise", "Starter"]

TEMPLATES = {
    "account_access": [
        ("calm", "I'm having trouble logging into {product}. It says my password is incorrect even though I'm sure it's right."),
        ("annoyed", "I've been locked out of {product} {time}. This is really inconvenient, can someone help?"),
        ("angry", "I am completely locked out of {product} {time} and I have work that depends on it. This needs to be fixed immediately."),
        ("calm", "My two-factor authentication code isn't being accepted on {product}. Can you help me regain access?"),
        ("annoyed", "{product} keeps logging me out every few minutes {time}, it's disrupting my work."),
        ("curious", "I forgot my password for {product} -- what's the process to reset it?"),
        ("angry", "I still cannot get into {product} {time} despite resetting my password twice. I need this resolved now."),
        ("calm", "I never received the verification email to activate my account on {product}."),
    ],
    "billing_problem": [
        ("annoyed", "I was charged {amount} on my last invoice for {product} but I downgraded my plan {time}."),
        ("angry", "I was billed twice this month for {product}, totaling an extra {amount}. Please refund this immediately."),
        ("calm", "Can someone explain the {amount} charge on my latest invoice for {product}?"),
        ("annoyed", "I downgraded from {plan} to a cheaper plan {time} but I'm still being billed the old rate for {product}."),
        ("curious", "I need to update my credit card on file for {product} billing -- where do I do that?"),
        ("angry", "There's a charge of {amount} on my card from {product} that I did not authorize. This needs to be investigated right away."),
        ("calm", "Could you send me a copy of my most recent invoice for {product}?"),
    ],
    "bug": [
        ("annoyed", "{product} is not saving my changes {time}. I make an edit and it just reverts."),
        ("angry", "{product} keeps crashing with a {error} error {time}. I've lost work because of this."),
        ("calm", "I noticed a small visual glitch in {product} where some text overlaps on smaller screens."),
        ("angry", "{product} is throwing a {error} error {time} and I cannot complete any of my tasks. This is blocking my entire team."),
        ("annoyed", "Several buttons in {product} stopped responding {time}, I have to refresh the page constantly."),
        ("calm", "When I try to sort results in {product}, the order doesn't seem to change."),
    ],
    "feature_request": [
        ("curious", "Is there any plan to add a dark mode to {product}? Would really improve my experience."),
        ("happy", "I really enjoy using {product}! It would be even better with the ability to bulk edit items."),
        ("calm", "It would be great if {product} supported exporting directly to Excel format."),
        ("curious", "Does {product} support keyboard shortcuts, or is that something you're considering adding?"),
        ("happy", "Loving {product} so far -- one thing that would help is a way to save custom filters."),
    ],
    "how_to": [
        ("curious", "I'm new here -- could you point me to a guide on how to set up {product}?"),
        ("calm", "I have a general question about how {product} handles permissions across team members."),
        ("happy", "Loving the product so far! Quick question -- how do I invite teammates to {product}?"),
        ("curious", "Is there documentation available for {product}? I'd like to learn more before diving in."),
        ("calm", "Could you walk me through how to change the default settings in {product}?"),
    ],
    "other": [
        ("calm", "I have a general question about my account and how {product} fits into my current plan."),
        ("curious", "We need details about your data encryption and compliance certifications for {product}."),
        ("calm", "I'm interested in exploring a partnership opportunity related to {product}. Who should I speak with?"),
        ("curious", "Just wanted to check if there's an update on my previous request regarding {product}."),
    ],
    "performance": [
        ("annoyed", "{product} has been noticeably slower {time}, pages take much longer to load than usual."),
        ("angry", "{product} has degraded so badly {time} that my entire team's work is at a standstill. We need this fixed urgently."),
        ("calm", "I've noticed occasional lag in {product}, nothing severe but worth mentioning."),
        ("annoyed", "{product} takes over 30 seconds to load {time}, it used to be instant."),
        ("angry", "{product} is essentially unusable {time} -- every action times out with a {error} error."),
    ],
    "security_concern": [
        ("angry", "I noticed a suspicious login on my account from an unrecognized location {time}. Please investigate immediately."),
        ("calm", "Can you confirm whether {product} encrypts data at rest and in transit?"),
        ("angry", "I think my account tied to {product} may have been compromised {time} -- I'm seeing activity I didn't perform."),
        ("annoyed", "I received an alert about a new device accessing {product} {time} that wasn't me."),
        ("calm", "Do you have a SOC 2 report or security audit documentation available for {product}?"),
    ],
}

TONE_TO_SENTIMENT = {"angry": "very_negative", "annoyed": "negative", "calm": "neutral",
                      "curious": "neutral", "happy": "positive"}
TONE_URGENCY_BONUS = {"angry": 4, "annoyed": 2, "calm": 0, "curious": -1, "happy": -2}
URGENT_KEYWORDS = ["immediately", "urgent", "right away", "as soon as possible",
                    "critical", "blocking", "entire team", "cannot", "now"]


def fill_slots(text, product_area):
    return text.format(
        product=random.choice(PRODUCTS_HUMAN[product_area]),
        time=random.choice(TIME_PHRASES),
        error=random.choice(ERROR_CODES),
        amount=random.choice(AMOUNTS),
        plan=random.choice(PLAN_NAMES),
    )


def sentiment_with_noise(base_sentiment):
    order = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    idx = order.index(base_sentiment)
    if random.random() < 0.12:
        idx = max(0, min(len(order) - 1, idx + random.choice([-1, 1])))
    return order[idx]


def compute_priority(issue_type, tone, segment, channel, has_attachment, text):
    score = ISSUE_SEVERITY[issue_type]
    score += TONE_URGENCY_BONUS[tone]
    score += SEGMENT_WEIGHT[segment]
    score += CHANNEL_WEIGHT[channel]
    score += 1.0 if has_attachment else 0.0

    lower = text.lower()
    score += sum(1 for kw in URGENT_KEYWORDS if kw in lower) * 0.7
    score += np.random.normal(0, 1.5)  # realistic judgment-call noise

    if score >= 14:
        return "urgent"
    elif score >= 10:
        return "high"
    elif score >= 6:
        return "medium"
    else:
        return "low"


def generate(n_rows=30000):
    rows = []
    for i in range(n_rows):
        issue_type = random.choice(ISSUE_TYPES)
        product_area = random.choice(PRODUCT_AREAS)
        tone, body_template = random.choice(TEMPLATES[issue_type])
        message = fill_slots(body_template, product_area)

        if random.random() < 0.25:
            extras = [
                " Let me know what you need from me.",
                " I've attached a screenshot for reference.",
                " This is affecting multiple users on my team.",
                " Happy to hop on a call if that's easier.",
                " Please advise on next steps.",
            ]
            message += random.choice(extras)

        segment = random.choices(CUSTOMER_SEGMENTS, weights=[0.35, 0.25, 0.15, 0.15, 0.10])[0]
        channel = random.choices(CHANNELS, weights=[0.40, 0.25, 0.10, 0.15, 0.10])[0]
        platform = random.choices(PLATFORMS, weights=[0.40, 0.20, 0.20, 0.10, 0.10])[0]
        region = random.choice(REGIONS)
        has_attachment = int(random.random() < 0.22)

        base_sentiment = TONE_TO_SENTIMENT[tone]
        sentiment = sentiment_with_noise(base_sentiment)
        priority = compute_priority(issue_type, tone, segment, channel, has_attachment, message)

        rows.append({
            "initial_message": message,
            "customer_segment": segment,
            "channel": channel,
            "product_area": product_area,
            "platform": platform,
            "region": region,
            "has_attachment": has_attachment,
            "customer_sentiment": sentiment,
            "issue_type": issue_type,
            "priority": priority,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate(30000)
    df.to_csv("synthetic_it_support_tickets.csv", index=False)
    print("Shape:", df.shape)
    print("Unique messages:", df["initial_message"].nunique())
    print()
    print(df["priority"].value_counts(normalize=True).round(3))
    print()
    print(pd.crosstab(df["issue_type"], df["priority"], normalize="index").round(3))
    print()
    print(pd.crosstab(df["customer_segment"], df["priority"], normalize="index").round(3))
