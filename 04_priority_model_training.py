# ==============================================
# Phase 7 & 8: Improved Priority Model Training
# ==============================================

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Improved Priority Model Training")
print("="*60)

# Load data
X = joblib.load("../X_features.pkl")
y_priority = joblib.load("../y_priority.pkl")

print("Priority Distribution:")
print(y_priority.value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ==============================================
# Models (Only stable ones)
# ==============================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        C=0.5,
        class_weight='balanced',
        solver='lbfgs'
    ),
    
    "Linear SVC": LinearSVC(
        C=0.5,
        class_weight='balanced',
        max_iter=3000,
        dual=False
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1": f1,
        "predictions": y_pred
    }

# Select best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print("\n" + "="*60)
print(f"Best Priority Model: {best_model_name}")
print(f"Accuracy : {results[best_model_name]['accuracy']:.4f}")
print(f"F1-Score : {results[best_model_name]['f1']:.4f}")
print("="*60)

# Save best model
joblib.dump(best_model, "../best_priority_model.pkl")
print("Best Priority model saved!")

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nPriority Model Training Completed!")