# ==============================================
# Phase 7 & 8: Improved Priority Model Training
# ==============================================
#
# CHANGE vs original: same GroupShuffleSplit fix as the issue_type script,
# plus train/test accuracy printed together so you can see the real gap.
# With the new dataset, priority now has genuine signal (issue severity +
# customer_segment + channel + has_attachment + urgency language), so
# accuracy should land meaningfully above the majority-class baseline
# (~35%, since "low" is ~35% of rows) instead of hovering near it.

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent

print("="*60)
print("Improved Priority Model Training")
print("="*60)

X = joblib.load(PROJECT_DIR / "X_features.pkl")
y_priority = joblib.load(PROJECT_DIR / "y_priority.pkl")
groups = joblib.load(PROJECT_DIR / "message_groups.pkl")

print("Priority Distribution:")
print(y_priority.value_counts(normalize=True).round(3))

# Group-aware split (same rationale as issue_type script)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_priority, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_priority.iloc[train_idx], y_priority.iloc[test_idx]

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Baseline: majority-class dummy classifier, so you know what "beating chance" means
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"\nMajority-class baseline accuracy: {dummy_acc:.4f}  <- any real model must clear this")

# ==============================================
# Models
# ==============================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500, C=0.5, class_weight='balanced', solver='lbfgs'
    ),
    "Linear SVC": LinearSVC(
        C=0.5, class_weight='balanced', max_iter=3000, dual=False
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_split=8,
        min_samples_leaf=3, class_weight='balanced_subsample',
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
    print(f"Test Accuracy  : {acc:.4f}  (baseline: {dummy_acc:.4f})")
    print(f"Overfit Gap    : {train_acc - acc:.4f}")
    print(f"Test F1-Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results[name] = {
        "model": model, "train_accuracy": train_acc, "accuracy": acc,
        "f1": f1, "predictions": y_pred
    }

best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print("\n" + "="*60)
print(f"Best Priority Model: {best_model_name}")
print(f"Accuracy : {results[best_model_name]['accuracy']:.4f}  (baseline: {dummy_acc:.4f})")
print(f"F1-Score : {results[best_model_name]['f1']:.4f}")
print("="*60)

joblib.dump(best_model, PROJECT_DIR / "best_priority_model.pkl")
print("Best Priority model saved!")

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nPriority Model Training Completed!")
