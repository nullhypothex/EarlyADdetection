import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# === Load Simulated Dataset ===
df = pd.read_csv("simulated_AD_users_groq_with_metrics.csv")
print("Original rows:", len(df))

# === Introduce Controlled Noise & Overlap ===

def add_feature_noise(row, noise_level=0.1):
    # Add random noise to numerical features
    row["watch_time_secs"] += np.random.normal(0, noise_level * 10)
    row["skipped_secs"] += np.random.normal(0, noise_level * 5)
    row["pause_count"] += np.random.randint(-1, 2)
    row["replay_count"] += np.random.randint(-1, 2)
    return row

def user_confounds(user_df):
    confound = random.choice(["slow_viewer", "impulsive", "low_liker"])
    if confound == "slow_viewer":
        user_df["watch_time_secs"] *= 0.8
        user_df["pause_count"] += 1
    elif confound == "impulsive":
        user_df["skipped_secs"] += 10
        user_df["replay_count"] += 1
    elif confound == "low_liker":
        user_df["liked"] = 0
        user_df["shared"] = 0
    return user_df

# Apply noise per row
df = df.apply(add_feature_noise, axis=1)

# Apply user-level confounds
for user in df["user_id"].unique():
    df.loc[df["user_id"] == user] = user_confounds(df[df["user_id"] == user])

# Clip values to realistic bounds
df["watch_time_secs"] = df["watch_time_secs"].clip(5, 70)
df["skipped_secs"] = df["skipped_secs"].clip(0, 30)
df["pause_count"] = df["pause_count"].clip(0, 6)
df["replay_count"] = df["replay_count"].clip(0, 5)

# === Encode Labels ===
label_map = {"Healthy": 0, "MCI": 1, "EarlyAD": 2}
df["target"] = df["label"].map(label_map)

# === Train/Val Split by Users ===
unique_users = df["user_id"].unique()
train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

train_df = df[df["user_id"].isin(train_users)]
val_df = df[df["user_id"].isin(val_users)]

print(f"Train users: {len(train_users)}, Validation users: {len(val_users)}")
# === Feature Columns ===
features = ["watch_time_secs", "skipped_secs", "pause_count", "replay_count", "liked", "shared"]
X_train = train_df[features]
y_train = train_df["target"]

X_val = val_df[features]
y_val = val_df["target"]

# === Train Model ===
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# === Save Model ===
joblib.dump(model, "alzheimers_model_noisy1.pkl")
print("Model saved as alzheimers_model_noisy1.pkl")

# === Evaluate ===
y_pred = model.predict(X_val)
print("\n Validation Classification Report:\n")
print(classification_report(y_val, y_pred, target_names=label_map.keys()))

# === Optional: Confusion Matrix Plot ===
def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.show()

# === Report ===
print("📊 Train Noisy Data Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_map.keys()))

# === Confusion matrix (optional) ===
cm = confusion_matrix(y_val, y_pred)
print("\n🔢 Confusion Matrix:")
print(cm)

