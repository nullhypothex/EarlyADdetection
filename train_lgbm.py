import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Load dataset ===
df = pd.read_csv("Lsimulated_AD_users_groq_metrics.csv")

# === Features and labels ===
features = [
    "coherence_score", "watch_time_secs", "skipped_secs",
    "pause_count", "replay_count", "liked", "shared"
]
X = df[features]
y = df["label"]

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Train LGBM ===
model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# === Predict ===
y_pred = model.predict(X_test)

# === Report ===
print(" LightGBM Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Confusion matrix (optional) ===
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:")
print(cm)
