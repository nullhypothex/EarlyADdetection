import requests
import random
import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Groq API Configuration ===
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# === SentenceTransformer for Coherence ===
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Video Categories and Descriptions ===
categories = ["Sports", "News", "Entertainment", "Health and Lifestyle", "Documentary"]
video_library = {
    "Sports": ["Match Highlights", "Goal Compilation"],
    "News": ["World Headlines", "Politics Roundup"],
    "Entertainment": ["Funny Cat", "Comedy Skit"],
    "Health and Lifestyle": ["Cooking Tips", "Yoga Basics"],
    "Documentary": ["Wildlife Documentary", "History of Medicine"]
}
video_descriptions = {
    "Funny Cat": "A humorous cat performing silly actions around the house.",
    "Comedy Skit": "A short comedy sketch performed by actors.",
    "Cooking Tips": "A chef demonstrates how to prepare a simple pasta dish.",
    "Yoga Basics": "A beginner yoga instructor guides through calming poses.",
    "Match Highlights": "Highlights from a recent competitive football match.",
    "Goal Compilation": "A series of impressive goals scored in soccer.",
    "World Headlines": "Top international stories from around the world.",
    "Politics Roundup": "Political developments in major global economies.",
    "Wildlife Documentary": "Animals in their natural habitat, hunting or interacting.",
    "History of Medicine": "How modern medicine evolved from ancient practices."
}

# === Global toggle for Groq API usage ===
USE_GROQ_API = True # Change to False if using Fallback. 

# === Compute Coherence ===
def compute_coherence(summary, baseline_vec):
    vec = sbert_model.encode([summary])[0]
    return float(cosine_similarity([vec], [baseline_vec])[0][0])

# === Load or Generate Baseline Summaries ===
def get_or_generate_baseline_summaries():
    path = "baseline_summaries.csv"
    if os.path.exists(path):
        print("📂 Using cached baseline summaries...")
        return pd.read_csv(path, index_col=0).to_dict()["summary"]

    print("🚀 Generating baseline summaries via Groq...")
    baseline = {}
    for video, desc in video_descriptions.items():
        fallback = f"The video is about {desc.lower()}"
        baseline[video] = fallback
        time.sleep(random.uniform(1.5, 3.5))

    pd.DataFrame.from_dict(baseline, orient="index", columns=["summary"]).to_csv(path)
    return baseline

# === Fallback Summary Generator ===
def fallback_summary_generator(video, label, day):
    base_prompt = video_descriptions[video]
    filler_phrases = [
        "Umm, I think?", "Then again, maybe not.", "Anyway...",
        "It was kind of odd.", "Sort of confusing.",
        "I forget a bit here.", "Not totally sure though."
    ]
    summary = f"This video was about {base_prompt.lower()}"
    sentences = summary.split('.')
    cleaned = [s.strip() for s in sentences if s.strip()]

    if label == "Healthy":
        final = ". ".join(cleaned)
    elif label == "MCI":
        final = ". ".join([
            s + " " + random.choice(filler_phrases) if random.random() < 0.4 else s
            for s in cleaned
        ])
    else:
        final = ". ".join([
            s + " " + random.choice(filler_phrases) if random.random() < 0.8 else s
            for s in cleaned
        ])
    return f"Day {day}: {final.strip()}."

# === Generate Summary with Groq or fallback ===
def generate_realistic_summary(video, label, day, retries=5):
    if not USE_GROQ_API:
        return fallback_summary_generator(video, label, day)

    base_prompt = video_descriptions[video]
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an assistant that summarizes videos in 1 to 5 coherent sentences."},
            {"role": "user", "content": f"Summarize this video: {base_prompt}"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    summary = None
    # for attempt in range(1, retries + 1):
    #     try:
    #         response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload, timeout=30)
    #         response.raise_for_status()
    #         summary = response.json()["choices"][0]["message"]["content"].strip()

    #         # 🚀 Throttle harder between Groq API calls
    #         time.sleep(random.uniform(5.0, 8.0))
    #         break

    #     except requests.exceptions.RequestException as e:
    #         wait = min((2 ** attempt) + random.uniform(2.0, 5.0), 60)  # ⏳ max wait capped at 60 sec
    #         print(f"⚠️ Groq API error on attempt {attempt}: {e}")
    #         print(f"⏳ Retrying in {round(wait, 2)} seconds...")
    #         time.sleep(wait)
    
    #     summary = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()

            # Longer pause to reduce API pressure
            time.sleep(random.uniform(5.0, 8.0))
            break  # ✅ Success, break the retry loop

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            if status == 503:
                wait = min((2 ** attempt) + random.uniform(5.0, 10.0), 90)
                print(f"🚨 Groq API error {status} on attempt {attempt}: {e}")
                print(f"⏳ Waiting {round(wait, 2)}s before retrying...")
            elif status == 429:
                wait = min((2 ** attempt) + random.uniform(3.0, 7.0), 60)
                print(f"⚠️ Groq API error {status} (Too Many Requests) on attempt {attempt}: {e}")
                print(f"⏳ Waiting {round(wait, 2)}s before retrying...")
            else:
                print(f"❌ Unhandled HTTP error {status} on attempt {attempt}: {e}")
                break  # stop retrying on unknown HTTP errors
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            wait = min((2 ** attempt) + random.uniform(3.0, 7.0), 60)
            print(f"⚠️ Groq API network error on attempt {attempt}: {e}")
            print(f"⏳ Retrying in {round(wait, 2)} seconds...")
            time.sleep(wait)

    if not summary:
        return fallback_summary_generator(video, label, day)

    filler_phrases = [
        "Umm, I think?", "Then again, maybe not.", "Anyway...",
        "It was kind of odd.", "Sort of confusing.",
        "I forget a bit here.", "Not totally sure though."
    ]
    sentences = summary.split('.')
    cleaned = [s.strip() for s in sentences if s.strip()]

    if label == "Healthy":
        final = ". ".join(cleaned)
    elif label == "MCI":
        final = ". ".join([
            s + " " + random.choice(filler_phrases) if random.random() < 0.4 else s
            for s in cleaned
        ])
    else:
        final = ". ".join([
            s + " " + random.choice(filler_phrases) if random.random() < 0.8 else s
            for s in cleaned
        ])

    return f"Day {day}: {final.strip()}."

# === Progressions ===
def determine_label_for_day(progression_type, day):
    """Assigns cognitive label based on user’s progression pattern."""
    if progression_type == "StableHealthy":
        return "Healthy"
    elif progression_type == "MildProgressor":
        return "Healthy" if day < 40 else "MCI"
    elif progression_type == "FastDecliner":
        return "MCI" if day < 30 else "EarlyAD"
    elif progression_type == "GradualDecliner":
        if day < 25:
            return "Healthy"
        elif day < 50:
            return "MCI"
        else:
            return "EarlyAD"
    elif progression_type == "StableMCI":
        return "MCI"
    else:  # StableEarlyAD
        return "EarlyAD"

# === Main Simulation ===
def simulate_users(start_uid, num_users=1, num_days=100, baseline_vectors=None):
    global USE_GROQ_API
    all_data = []

    baseline_summaries = get_or_generate_baseline_summaries()
    if baseline_vectors is None:
        baseline_vectors = {
            v: sbert_model.encode([s])[0] for v, s in baseline_summaries.items()
        }

    for uid in tqdm(range(start_uid, start_uid + num_users), desc=f"Batch Users {start_uid}-{start_uid + num_users - 1}"):
        user_id = f"User{uid:02d}"
        # if uid > 10:
        #     USE_GROQ_API = False

        # Assign a realistic progression pattern
        progression_type = random.choice([
            "StableHealthy",
            "MildProgressor",
            "FastDecliner",
            "GradualDecliner",
            "StableMCI",
            "StableEarlyAD"
        ])
        print(f"[{user_id}] Progression = {progression_type}")

        for day in range(1, num_days + 1):
            date = (datetime(2025, 1, 1) + timedelta(days=day - 1)).strftime("%Y-%m-%d")
            cat1, cat2 = random.sample(categories, 2)
            video_pool = list(set(video_library[cat1] + video_library[cat2]))
            video_titles = random.sample(video_pool, min(5, len(video_pool)))

            for video_title in video_titles:
                label = determine_label_for_day(progression_type, day)
                summary = generate_realistic_summary(video_title, label, day)
                coherence = compute_coherence(summary, baseline_vectors[video_title])

                if label == "Healthy":
                    watch_time = random.randint(45, 60)
                    skips = random.randint(0, 5)
                    pauses = random.randint(0, 2)
                    replays = random.randint(0, 1)
                    liked = 1 if random.random() < 0.7 else 0
                    shared = 1 if random.random() < 0.4 else 0

                elif label == "MCI":
                    watch_time = random.randint(30, 50)
                    skips = random.randint(5, 15)
                    pauses = random.randint(1, 3)
                    replays = random.randint(1, 2)
                    liked = 1 if random.random() < 0.5 else 0
                    shared = 1 if random.random() < 0.3 else 0

                else:  # Early AD
                    watch_time = random.randint(15, 35)
                    skips = random.randint(10, 25)
                    pauses = random.randint(2, 5)
                    replays = random.randint(2, 4)
                    liked = 1 if random.random() < 0.2 else 0
                    shared = 1 if random.random() < 0.1 else 0

                all_data.append({
                    "user_id": user_id,
                    "date": date,
                    "day": day,
                    "category_1": cat1,
                    "category_2": cat2,
                    "video_title": video_title,
                    "summary": summary,
                    "coherence_score": round(coherence, 3),
                    "watch_time_secs": watch_time,
                    "skipped_secs": skips,
                    "pause_count": pauses,
                    "replay_count": replays,
                    "liked": liked,
                    "shared": shared,
                    "label": label
                })

            time.sleep(random.uniform(0.8, 2.0))  # Between days

    df = pd.DataFrame(all_data)
    path = "Lsimulated_AD_users_groq.csv"
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved to {path}")

# === Execution === w/o Groq
'''
if __name__ == "__main__":
    print("Precomputing baseline summaries and embeddings...")
    base_summaries = get_or_generate_baseline_summaries()
    baseline_vectors = {v: sbert_model.encode([s])[0] for v, s in base_summaries.items()}

    # Avoid duplication
    existing_ids = set()
    path = "Lsimulated_AD_users_groq.csv"
    if os.path.exists(path):
        existing_df = pd.read_csv(path)
        existing_ids = set(existing_df["user_id"].unique())

    # Simulate up to 50 users
    for i in range(1, 51):
        uid = f"User{i:02d}"
        if uid in existing_ids:
            print(f"⏩ Skipping {uid} (already simulated)")
            continue

        try:
            simulate_users(start_uid=i, num_users=1, num_days=60, baseline_vectors=baseline_vectors)
        except Exception as e:
            print(f"❌ Error for {uid}: {e}")
            continue

        time.sleep(random.uniform(5, 10))
'''
# === Execution === w/ Groq
# if __name__ == "__main__":
#     print("✅ Groq API enabled. Simulating Users 51–60 using Groq...")

#     USE_GROQ_API = True  # Enable Groq summaries

#     # Precompute baseline embeddings only once
#     print("Precomputing baseline summaries and embeddings...")
#     base_summaries = get_or_generate_baseline_summaries()
#     baseline_vectors = {
#         v: sbert_model.encode([s])[0] for v, s in base_summaries.items()
#     }

#     # Simulate users 51–60 using Groq API
#     for i in range(51, 61):
#         simulate_users(
#             start_uid=i,
#             num_users=1,
#             num_days=60,
#             baseline_vectors=baseline_vectors
#         )
#         time.sleep(random.uniform(5, 10))  # cooldown between users

# if __name__ == "__main__":
#     print("🧠 Simulating 100 users over 100 days...")

#     # Randomly choose 20–25 users to simulate with Groq API
#     groq_user_ids = set(random.sample(range(1, 101), random.randint(20, 25)))
#     print(f"🎯 Users using Groq API: {sorted(groq_user_ids)}")

#     # Precompute baseline summaries and vectors once
#     print("🔍 Precomputing baseline summaries and embeddings...")
#     base_summaries = get_or_generate_baseline_summaries()
#     baseline_vectors = {
#         v: sbert_model.encode([s])[0] for v, s in base_summaries.items()
#     }

#     # Loop through 100 users
#     for i in range(1, 101):
#         USE_GROQ_API = i in groq_user_ids  # toggle Groq per user

#         print(f"\n🚦 Simulating User{i:02d} with {'Groq' if USE_GROQ_API else 'Fallback'}...")
#         try:
#             simulate_users(
#                 start_uid=i,
#                 num_users=1,
#                 num_days=100,  # updated to 100 days
#                 baseline_vectors=baseline_vectors
#             )
#         except Exception as e:
#             print(f"❌ Error for User{i:02d}: {e}")
#             continue

#         time.sleep(random.uniform(5, 10))  # cooldown between users

if __name__ == "__main__":
    print("🧠 Simulating Users 42 to 100 over 100 days...")

    # === Already completed users ===
    existing_users = set(range(1, 42))
    groq_users_completed = {1, 4, 7, 12, 15, 19, 22, 23, 31, 39, 41}
    total_groq_needed = random.randint(20, 25)
    groq_remaining_count = total_groq_needed - len(groq_users_completed)

    # === Eligible new users (42–100) ===
    new_users = list(range(42, 101))

    # === Randomly assign Groq API to a few more new users ===
    groq_users_new = set(random.sample(new_users, groq_remaining_count))
    groq_users_all = groq_users_completed.union(groq_users_new)

    print(f"🎯 New users using Groq API: {sorted(groq_users_new)}")
    print(f"✅ Total Groq users: {sorted(groq_users_all)}")

    # === Precompute baseline summaries and embeddings ===
    print("🔍 Precomputing baseline summaries and embeddings...")
    base_summaries = get_or_generate_baseline_summaries()
    baseline_vectors = {
        v: sbert_model.encode([s])[0] for v, s in base_summaries.items()
    }

    # === Simulate users 42–100 ===
    for i in new_users:
        USE_GROQ_API = i in groq_users_all

        print(f"\n🚦 Simulating User{i:02d} with {'Groq' if USE_GROQ_API else 'Fallback'}...")

        try:
            simulate_users(
                start_uid=i,
                num_users=1,
                num_days=100,
                baseline_vectors=baseline_vectors
            )
        except Exception as e:
            print(f"❌ Error for User{i:02d}: {e}")
            continue

        # 🧊 Cooldown after Groq user
        if USE_GROQ_API:
            print("🧊 Cooling down after Groq user to avoid 429 errors...")
            time.sleep(random.uniform(60, 90))
        else:
            time.sleep(random.uniform(5, 10))

// ✅ Total Groq users should be: [1, 4, 7, 12, 15, 19, 22, 23, 31, 39, 41, 65, 68, 72, 75, 78, 84, 87, 90, 94, 97, 100]



