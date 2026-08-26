# ==============================================
# Phase 5: Train-Test Split
# Phase 6: Model Building (Issue Type) - Anti Overfitting Version
# ==============================================
#
# CHANGE vs original: GroupShuffleSplit keyed on message text instead of
# train_test_split. This guarantees rows with identical cleaned_message
# never appear in both train and test -- the actual cause of the
# "overfitting" you were seeing (it was really train/test leakage from
# templated/duplicated text landing on both sides of a random split).
# We also now print TRAIN accuracy next to TEST accuracy so a real
# overfit gap (if any remains) is visible immediately.

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent

print("="*60)
print("Phase 5 & 6: Anti-Overfitting Model Training (Issue Type)")
print("="*60)

# Load processed data
X = joblib.load(PROJECT_DIR / "X_features.pkl")
y_issue = joblib.load(PROJECT_DIR / "y_issue.pkl")
groups = joblib.load(PROJECT_DIR / "message_groups.pkl")

print("Features shape:", X.shape)
print("Unique message groups:", groups.nunique(), "/", len(groups), "rows")

# ==============================================
# Phase 5: Group-Aware Train-Test Split
# ==============================================

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y_issue, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_issue.iloc[train_idx], y_issue.iloc[test_idx]

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
print(f"Message-text overlap between train/test: {len(overlap)} (should be 0)")

# ==============================================
# Phase 6: Strong Regularized Models
# ==============================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=0.5, class_weight='balanced', solver='lbfgs'
    ),
    "Linear SVC": LinearSVC(
        C=0.5, class_weight='balanced', max_iter=3000, dual=False
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {acc:.4f}")
    print(f"Overfit Gap    : {train_acc - acc:.4f}  (>0.10 is a warning sign)")
    print(f"Test F1-Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results[name] = {
        "model": model, "train_accuracy": train_acc, "test_accuracy": acc,
        "test_f1": f1, "predictions": y_pred
    }

# Select Best Model by test F1 (not train score)
best_model_name = max(results, key=lambda x: results[x]['test_f1'])
best_model = results[best_model_name]['model']

print("\n" + "="*60)
print(f"Best Model: {best_model_name}")
print(f"Train Accuracy : {results[best_model_name]['train_accuracy']:.4f}")
print(f"Test Accuracy  : {results[best_model_name]['test_accuracy']:.4f}")
print(f"Test F1-Score  : {results[best_model_name]['test_f1']:.4f}")
print("="*60)

joblib.dump(best_model, PROJECT_DIR / "best_issue_type_model.pkl")
print("Best Issue Type model saved!")

plt.figure(figsize=(10,7))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nPhase 5 and Phase 6 Completed (Anti-Overfitting Version)!")
