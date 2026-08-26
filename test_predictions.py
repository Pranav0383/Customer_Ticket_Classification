"""
Sanity-check test cases for the trained issue_type and priority models.

Run this from your project folder (same place as best_issue_type_model.pkl etc):
    python test_predictions.py

Each case is written in DIFFERENT wording than the training templates
(paraphrased, not copy-pasted) so this also tests generalization, not
just memorization of the training text.

`expected_issue_type` should match closely -- issue_type is fairly
unambiguous from wording.

`expected_priority_band` is NOT a single exact label -- priority has
randomness baked into the training data by design (real triage involves
judgment calls), so treat this as "should land in this band or one step
either side of it", not an exact match requirement. What matters most is
the DIRECTION: cases 1/5/7 should clearly score higher than cases 2/4/8.
"""

from pathlib import Path
import importlib.util

# 05_prediction_system.py starts with a digit, so it can't be imported with
# a normal `import` statement -- load it directly from its file path instead.
_module_path = Path(__file__).resolve().parent / "05_prediction_system.py"
_spec = importlib.util.spec_from_file_location("prediction_system", _module_path)
pred_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pred_module)

predict_ticket = pred_module.predict_ticket

TEST_CASES = [
    dict(
        label="1. Severe security incident, enterprise, phone",
        initial_message="Someone accessed our enterprise account from an unknown device overnight and changed settings we didn't authorize. This is a serious security breach and we need it investigated right now.",
        customer_tier="enterprise", channel="phone_transcript", product_area="login_auth",
        platform="web", region="NA", customer_sentiment="very_negative", has_attachment=1,
        expected_issue_type="security_concern", expected_priority_band="urgent/high",
    ),
    dict(
        label="2. Casual how-to question, individual, email",
        initial_message="Hi, just wondering how I can invite a friend to join my account on the mobile app?",
        customer_tier="individual", channel="email", product_area="mobile_app",
        platform="android", region="APAC", customer_sentiment="positive", has_attachment=0,
        expected_issue_type="how_to", expected_priority_band="low",
    ),
    dict(
        label="3. Double billing, small business, chat",
        initial_message="I was billed twice for my subscription this month and it's an extra $120 I wasn't expecting. Can you refund the duplicate charge?",
        customer_tier="small_business", channel="chat", product_area="billing",
        platform="web", region="EU", customer_sentiment="negative", has_attachment=0,
        expected_issue_type="billing_problem", expected_priority_band="medium/high",
    ),
    dict(
        label="4. Feature suggestion, non-profit, web form",
        initial_message="Would it be possible to add a dark mode to the dashboard sometime? Not urgent, just a nice-to-have suggestion.",
        customer_tier="non_profit", channel="web_form", product_area="analytics_dashboard",
        platform="web", region="NA", customer_sentiment="neutral", has_attachment=0,
        expected_issue_type="feature_request", expected_priority_band="low",
    ),
    dict(
        label="5. Critical bug blocking whole team, enterprise, phone, attachment",
        initial_message="The API integration is throwing a 500 error on every request since this morning and it's completely blocking our entire engineering team. We need this fixed immediately.",
        customer_tier="enterprise", channel="phone_transcript", product_area="api_integration",
        platform="api_client", region="NA", customer_sentiment="very_negative", has_attachment=1,
        expected_issue_type="bug", expected_priority_band="urgent/high",
    ),
    dict(
        label="6. Simple account lockout, individual, email",
        initial_message="I can't log into my account, it keeps saying my password is wrong even though I just reset it.",
        customer_tier="individual", channel="email", product_area="login_auth",
        platform="ios", region="LATAM", customer_sentiment="neutral", has_attachment=0,
        expected_issue_type="account_access", expected_priority_band="low/medium",
    ),
    dict(
        label="7. Severe performance degradation, enterprise, chat, attachment",
        initial_message="Our dashboard has been essentially unusable since yesterday, every action times out. This is affecting our whole team's ability to work and we need urgent help.",
        customer_tier="enterprise", channel="chat", product_area="analytics_dashboard",
        platform="web", region="EU", customer_sentiment="very_negative", has_attachment=1,
        expected_issue_type="performance", expected_priority_band="urgent/high",
    ),
    dict(
        label="8. General compliance question, education, email",
        initial_message="We're evaluating your platform for our school and need to know about your data encryption and compliance certifications.",
        customer_tier="education", channel="email", product_area="data_export",
        platform="web", region="NA", customer_sentiment="neutral", has_attachment=0,
        expected_issue_type="other", expected_priority_band="low/medium",
    ),
]

print(f"{'Case':<55}{'Pred Issue':<20}{'Expected':<20}{'Pred Priority':<15}{'Expected Band':<15}")
print("-" * 125)

correct_issue = 0
for case in TEST_CASES:
    result = predict_ticket(
        initial_message=case["initial_message"],
        customer_tier=case["customer_tier"],
        channel=case["channel"],
        product_area=case["product_area"],
        platform=case["platform"],
        region=case["region"],
        customer_sentiment=case["customer_sentiment"],
        has_attachment=case["has_attachment"],
    )
    pred_issue = result["Predicted Issue Type"]
    pred_priority = result["Predicted Priority"]
    match = "OK" if pred_issue == case["expected_issue_type"] else "CHECK"
    if pred_issue == case["expected_issue_type"]:
        correct_issue += 1

    print(f"{case['label']:<55}{pred_issue:<20}{case['expected_issue_type']:<20}{pred_priority:<15}{case['expected_priority_band']:<15}  [{match}]")

print("-" * 125)
print(f"\nIssue type matched expectation on {correct_issue}/{len(TEST_CASES)} cases.")
print("For priority: check that urgent/high-labeled cases (1,5,7) score visibly")
print("higher than the low-labeled cases (2,4) -- exact label match isn't required.")
