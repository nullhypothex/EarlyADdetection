import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# Load dataset
df = pd.read_csv("simulated_AD_users_groq.csv")
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['day'].astype(int)
df = df.sort_values(['user_id', 'date'])

# Output setup
os.makedirs("coherence_plots_labeled", exist_ok=True)
pdf_path = "coherence_plots_labeled/User_Coherence_Trends_With_Labels1.pdf"
pdf = PdfPages(pdf_path)

users = df['user_id'].unique()

for user in users:
    user_df = df[df['user_id'] == user]
    daily_avg = user_df.groupby(['day', 'label'])['coherence_score'].mean().reset_index()

    # Working memory baseline
    baseline = daily_avg[daily_avg['day'] <= 5]['coherence_score'].mean()
    daily_avg['delta'] = daily_avg['coherence_score'] - baseline
    daily_avg['flagged'] = daily_avg['delta'] <= -0.1

    # Set up label background colors
    label_map = {'Healthy': '#d0f0c0', 'MCI': '#f9f1a5', 'EarlyAD': '#f4cccc'}
    day_labels = user_df.groupby('day')['label'].first()

    plt.figure(figsize=(10, 5))

    # Background shading
    for label, color in label_map.items():
        day_ranges = day_labels[day_labels == label].index
        if not day_ranges.empty:
            plt.axvspan(day_ranges.min(), day_ranges.max(), color=color, alpha=0.2, label=f'{label} Phase')

    # Plot coherence and thresholds
    sns.lineplot(data=daily_avg, x='day', y='coherence_score', marker='o', label='Avg Coherence', color='black')
    plt.axhline(0.85, color='green', linestyle='--', label='Global Healthy Threshold')
    plt.axhline(0.5, color='red', linestyle='--', label='Global EarlyAD Threshold')
    plt.axhline(baseline, color='blue', linestyle='--', label=f'{user} Baseline ({baseline:.2f})')

    flagged_days = daily_avg[daily_avg['flagged']]
    plt.scatter(flagged_days['day'], flagged_days['coherence_score'], color='red', s=60, label='≥10% Drop')

    plt.title(f"{user} — Coherence Drift with Cognitive Labels")
    plt.xlabel("Day")
    plt.ylabel("Average Coherence Score")
    plt.ylim(0.3, 1.0)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

pdf.close()
print(f"✅ Saved all user plots to: {pdf_path}")
