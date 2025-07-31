# Simple retraining script
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

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

gb_tuned = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_tuned.fit(X_train, y_train)
joblib.dump(gb_tuned, "ml/gradient_boost_tuned.pkl")

print("Models retrained and saved!")