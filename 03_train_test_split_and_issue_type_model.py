# ==============================================
# Phase 5: Train-Test Split
# Phase 6: Model Building (Issue Type) - Anti Overfitting Version
# ==============================================

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Phase 5 & 6: Anti-Overfitting Model Training (Issue Type)")
print("="*60)

# Load processed data
X = joblib.load("X_features.pkl")
y_issue = joblib.load("y_issue.pkl")

print("Features shape:", X.shape)

# ==============================================
# Phase 5: Train-Test Split
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_issue,
    test_size=0.25,
    random_state=42,
    stratify=y_issue
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ==============================================
# Phase 6: Strong Regularized Models
# ==============================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=0.01,                      # Strong regularization
        class_weight='balanced',
        solver='lbfgs'               # Changed from liblinear
    ),
    
    "Linear SVC": LinearSVC(
        C=0.01,
        class_weight='balanced',
        max_iter=3000,
        dual=False
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    
    # Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Cross-Validation F1 Scores: {cv_scores}")
    print(f"Average CV F1-Score: {cv_scores.mean():.4f}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test F1-Score : {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        "model": model,
        "cv_f1": cv_scores.mean(),
        "test_accuracy": acc,
        "test_f1": f1,
        "predictions": y_pred
    }

# Select Best Model
best_model_name = max(results, key=lambda x: results[x]['cv_f1'])
best_model = results[best_model_name]['model']

print("\n" + "="*60)
print(f"Best Model: {best_model_name}")
print(f"Cross-Validation F1 : {results[best_model_name]['cv_f1']:.4f}")
print(f"Test Accuracy       : {results[best_model_name]['test_accuracy']:.4f}")
print(f"Test F1-Score       : {results[best_model_name]['test_f1']:.4f}")
print("="*60)

# Save best model
joblib.dump(best_model, "best_issue_type_model.pkl")
print("Best Issue Type model saved!")

# Confusion Matrix
plt.figure(figsize=(10,7))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nPhase 5 and Phase 6 Completed (Anti-Overfitting Version)!")