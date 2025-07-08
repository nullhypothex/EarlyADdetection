import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import seaborn as sns

# Load simulated data
df = pd.read_csv("simulated_AD_users_groq.csv")
df["label_encoded"] = df["label"].map({"Healthy": 0, "MCI": 1, "EarlyAD": 2})
df.sort_values(by=["user_id", "day"], inplace=True)

# Feature and target
features = ["watch_time_secs", "skipped_secs", "pause_count", "replay_count", "liked", "shared", "coherence_score"]
X = df[features].copy()
y = df["label_encoded"]
groups = df["user_id"]

# GroupKFold setup
gkf = GroupKFold(n_splits=5)
all_reports = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    # Add time-aware feature: coherence drift (rolling difference)
    for X_df in [X_train, X_val]:
        X_df["coherence_drift"] = df.loc[X_df.index].groupby("user_id")["coherence_score"].transform(
            lambda x: x - x.rolling(window=3, min_periods=1).mean().shift(1).fillna(0)
        )

    # Semi-supervised noise: flip 5% of training labels
    y_train_noisy = y_train.copy()
    noise_mask = np.random.rand(len(y_train_noisy)) < 0.05
    y_train_noisy[noise_mask] = np.random.randint(0, 3, size=noise_mask.sum())

    # Build and train pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000))
    ])
    model.fit(X_train, y_train_noisy)

    # Evaluate on validation
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=["Healthy", "MCI", "EarlyAD"], output_dict=True)
    all_reports.append(report)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "MCI", "EarlyAD"],
                yticklabels=["Healthy", "MCI", "EarlyAD"])
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.savefig(f"confusion_matrix_fold{fold+1}.png")
    plt.clf()

# Save final model
model.fit(X, y)
joblib.dump(model, "alzheimers_model_timeaware.pkl")
print(" Saved model as alzheimers_model_timeaware.pkl")

# Save cross-validation average report
report_df = pd.DataFrame([r["weighted avg"] for r in all_reports])
report_df.to_csv("crossval_avg_report.csv", index=False)
print("📊 Cross-validation average report saved.")
