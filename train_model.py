import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# === Load the dataset ===
df = pd.read_csv("simulated_AD_users_groq_with_metrics.csv")

# === Select Features and Labels ===
features = [
    "coherence_score", "watch_time_secs", "skipped_secs",
    "pause_count", "replay_count", "liked", "shared"
]
X = df[features]
y = df["label"]

# === Encode Labels ===
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Train Logistic Regression Model ===
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# === Predict on Test Set ===
y_pred = model.predict(X_test)

# === Evaluate ===
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=False)
print("📊 Classification Report:")
print(report)

# === Save Model (Optional) ===
with open("alzheimers_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as alzheimers_classifier.pkl")

# === Save Report (Optional) ===
from sklearn.metrics import classification_report
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv("classification_report.csv")
print("📁 Report saved to classification_report.csv")

# === Confusion Matrix (Optional) ===
cm = confusion_matrix(y_test, y_pred)
print("\n🔢 Confusion Matrix:")
print(cm)
