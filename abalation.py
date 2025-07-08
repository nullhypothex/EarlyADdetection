import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# === Load the dataset ===
df = pd.read_csv("simulated_AD_users_groq_with_metrics.csv")

def save_classification_report(y_true, y_pred, use_coherence, use_behavior):
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.3f', cbar=False, ax=ax)
    ax.set_title(f"Classification Report\nBehavior={use_behavior}, Coherence={use_coherence}")
    plt.tight_layout()
    plt.savefig(f"classification_report_Behavior={use_behavior}_Coherence={use_coherence}.png")
    plt.close(fig)

def save_confusion_matrix(y_true, y_pred, use_coherence, use_behavior):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix\nBehavior={use_behavior}, Coherence={use_coherence}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_Behavior={use_behavior}_Coherence={use_coherence}.png")
    plt.close(fig)

def run_ablation_model(df, use_coherence=True, use_behavior=True, label_col='label'):
    features = []
    if use_behavior:
        features += ['watch_time_secs','skipped_secs','pause_count','replay_count','liked','shared']
    if use_coherence:
        features += ['coherence_score', 'BLEU','ROUGE_L','embedding_similarity']

    X = df[features]
    y = df[label_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    print(f"\n✅ Setting: Behavior={use_behavior}, Coherence={use_coherence}")
    print(classification_report(y, y_pred, digits=3))
    
    # Save classification report and confusion matrix as figures
    save_classification_report(y, y_pred, use_coherence, use_behavior)
    save_confusion_matrix(y, y_pred, use_coherence, use_behavior)

# Run different ablation settings
print("🔬 FULL MODEL")
run_ablation_model(df, use_coherence=True, use_behavior=True)

print("\n❌ NO COHERENCE")
run_ablation_model(df, use_coherence=False, use_behavior=True)

print("\n❌ BEHAVIOR ONLY")
run_ablation_model(df, use_coherence=False, use_behavior=True)

print("\n❌ NO NOISE (Train on Clean Data)")
df_clean = pd.read_csv("simulated_AD_users_groq.csv")
run_ablation_model(df_clean, use_coherence=True, use_behavior=True)
