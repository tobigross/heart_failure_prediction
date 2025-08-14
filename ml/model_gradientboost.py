# Simple retraining script
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 8)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42
    )

    # Cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score
# Load and prepare data (same as your existing logic)
data = pd.read_csv("data/heart.csv")
data_encoded = pd.get_dummies(data.drop(columns=["HeartDisease"]))

    # Load training columns
with open("ml/training_columns.txt") as f:
    training_columns = [line.strip() for line in f if line.strip()]

X = data_encoded.reindex(columns=training_columns, fill_value=0).astype(float)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train standard sklearn models (no custom classes)
gb_default = GradientBoostingClassifier(random_state=42)
gb_default.fit(X_train, y_train)
joblib.dump(gb_default, "ml/gradient_boost_default.pkl")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_trial.params
print("Best params after tuning: ",best_params)
gb_tuned = GradientBoostingClassifier(**best_params, random_state=42)
gb_tuned.fit(X_train, y_train)
joblib.dump(gb_tuned, "ml/gradient_boost_tuned.pkl")
y_pred_untuned = gb_default.predict(X_test)
y_pred_tuned   = gb_tuned.predict(X_test)

print("Models retrained and saved!")
print("Test Accuracy untuned:", gb_default.score(X_test, y_test))
print("Test Accuracy tuned :", gb_tuned.score(X_test, y_test))
print("Untuned Model Metrics:")
print(classification_report(y_test, y_pred_untuned, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_untuned))
print("ROC-AUC:", roc_auc_score(y_test, gb_default.predict_proba(X_test)[:, 1]))

print("\nTuned Model Metrics:")
print(classification_report(y_test, y_pred_tuned, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
print("ROC-AUC:", roc_auc_score(y_test, gb_tuned.predict_proba(X_test)[:, 1]))
#feature importance
importances = gb_tuned.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.ylabel("Feature Importance")
plt.title("Tuned Gradient Boosting Feature Importance")
plt.tight_layout()
plt.show()
