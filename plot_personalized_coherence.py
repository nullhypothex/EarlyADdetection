import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load your dataset ===
df = pd.read_csv("simulated_AD_users_groq.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['user_id', 'date'])

# Output folder
os.makedirs("coherence_plots_labeled", exist_ok=True)

# Get list of users
users = df['user_id'].unique()

for user in users:
    user_df = df[df['user_id'] == user].copy()
    user_df['day'] = user_df['day'].astype(int)

    # Daily average coherence
    daily = user_df.groupby('day')['coherence_score'].mean().reset_index()

    # Compute personal working memory baseline (first 5 days)
    personal_baseline = daily[daily['day'] <= 5]['coherence_score'].mean()

    # === Plot ===
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=daily, x='day', y='coherence_score', marker='o', linewidth=2.5)

    plt.axhline(y=0.85, color='green', linestyle='--', label='Healthy Threshold (Global)')
    plt.axhline(y=0.5, color='red', linestyle='--', label='EarlyAD Threshold (Global)')
    plt.axhline(y=personal_baseline, color='blue', linestyle='--', label=f'{user} Baseline')

    plt.title(f"{user} — Coherence Drift w/ Working Memory Benchmark", fontsize=14)
    plt.xlabel("Day")
    plt.ylabel("Avg Coherence Score")
    plt.legend()
    plt.tight_layout()

    # Save
    plt.savefig(f"coherence_plots_labeled/{user}_personalized_trend.png")
    plt.close()

print("✅ Saved all personalized coherence plots with working memory baselines.")