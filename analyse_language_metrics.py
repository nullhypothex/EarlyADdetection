import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("simulated_AD_users_groq.csv")

# Load SentenceTransformer model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Compute working memory baseline: Day 1–5 summary per user+video
baseline_df = df[df["day"] <= 5].groupby(["user_id", "video_title"])["summary"].first().reset_index()
baseline_map = {(row["user_id"], row["video_title"]): row["summary"] for _, row in baseline_df.iterrows()}

# Initialize lists
bleu_scores = []
rouge_l_scores = []
embedding_similarities = []

# Scorer configs
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smoothie = SmoothingFunction().method4

# Compute metrics
for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing summaries"):
    user = row["user_id"]
    video = row["video_title"]
    curr_summary = row["summary"]
    baseline_summary = baseline_map.get((user, video), "")

    # BLEU
    bleu = sentence_bleu([baseline_summary.split()], curr_summary.split(), smoothing_function=smoothie) if baseline_summary else 0.0
    bleu_scores.append(bleu)

    # ROUGE-L
    rouge = scorer.score(baseline_summary, curr_summary)["rougeL"].fmeasure if baseline_summary else 0.0
    rouge_l_scores.append(rouge)

    # SBERT Embedding Similarity
    try:
        emb1 = sbert.encode(curr_summary, convert_to_tensor=True)
        emb2 = sbert.encode(baseline_summary, convert_to_tensor=True)
        emb_sim = util.pytorch_cos_sim(emb1, emb2).item()
    except:
        emb_sim = 0.0
    embedding_similarities.append(emb_sim)

# Append results
df["BLEU"] = bleu_scores
df["ROUGE_L"] = rouge_l_scores
df["embedding_similarity"] = embedding_similarities

# Save updated file
df.to_csv("simulated_AD_users_groq_with_metrics.csv", index=False)
print("✅ Metrics saved to 'simulated_AD_users_groq_with_metrics.csv'")

# Summary stats
print("\n📊 Average Scores by Label:")
print(df.groupby("label")[["BLEU", "ROUGE_L", "embedding_similarity"]].mean().round(3))

# Optional: Plot distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x="label", y="embedding_similarity", data=df)
plt.title("Embedding Similarity Across Labels")

plt.savefig(f"embedding_similarity1.png")
plt.close()
