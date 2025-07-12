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
GROQ_API_KEY = "Enter_your_Groq_API"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# === SentenceTransformer for Coherence ===
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Video Data ===
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

USE_GROQ_API = True

def compute_coherence(summary, baseline_vec):
    vec = sbert_model.encode([summary])[0]
    return float(cosine_similarity([vec], [baseline_vec])[0][0])

def get_or_generate_baseline_summaries():
    path = "baseline_summaries.csv"
    if os.path.exists(path):
        print("📂 Using cached baseline summaries...")
        return pd.read_csv(path, index_col=0).to_dict()["summary"]

    print("🚀 Generating baseline summaries via fallback...")
    baseline = {}
    for video, desc in video_descriptions.items():
        fallback = f"The video is about {desc.lower()}"
        baseline[video] = fallback
        time.sleep(random.uniform(1.5, 3.5))

    pd.DataFrame.from_dict(baseline, orient="index", columns=["summary"]).to_csv(path)
    return baseline

def fallback_summary_generator(video, label, day):
    base_prompt = video_descriptions[video]
    summary = f"This video was about {base_prompt.lower()}"
    return add_label_noise(summary, label, day)


def generate_realistic_summary(video, label, day, max_retries=8):
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
    for attempt in range(1, max_retries + 1):
        try:
            wait_before = random.uniform(18.0, 28.0)  # throttle before call
            time.sleep(wait_before)

            response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload, timeout=45)
            if response.status_code == 429:
                backoff = min(30 + attempt * random.uniform(10, 20), 120)
                print(f"⚠️ 429 Too Many Requests — backing off for {backoff:.2f}s (Attempt {attempt})")
                time.sleep(backoff)
                continue
            elif response.status_code == 503:
                backoff = min(20 + attempt * random.uniform(5, 15), 90)
                print(f"⚠️ 503 Service Unavailable — waiting {backoff:.2f}s (Attempt {attempt})")
                time.sleep(backoff)
                continue

            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()
            return add_label_noise(summary, label, day)

        except requests.exceptions.RequestException as e:
            wait = min(60 + attempt * random.uniform(10, 30), 180)
            print(f"⚠️ Groq Request Exception: {e} — retrying in {wait:.2f}s (Attempt {attempt})")
            time.sleep(wait)

    print("❌ Max retries reached. Using fallback.")
    return fallback_summary_generator(video, label, day)

def add_label_noise(summary, label, day):
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

def determine_label_for_day(progression_type, day):
    if progression_type == "StableHealthy":
        return "Healthy"
    elif progression_type == "MildProgressor":
        return "Healthy" if day < 70 else "MCI"
    elif progression_type == "FastDecliner":
        return "MCI" if day < 60 else "EarlyAD"
    elif progression_type == "GradualDecliner":
        if day < 42:
            return "Healthy"
        elif day < 80:
            return "MCI"
        else:
            return "EarlyAD"
    elif progression_type == "StableMCI":
        return "MCI"
    else:
        return "EarlyAD"
    
def simulate_users(start_uid, num_users=1, num_days=100, baseline_vectors=None):
    global USE_GROQ_API
    path = "Lsimulated_AD_users_groq.csv"

    baseline_summaries = get_or_generate_baseline_summaries()
    if baseline_vectors is None:
        baseline_vectors = {
            v: sbert_model.encode([s])[0] for v, s in baseline_summaries.items()
        }

    # === Detect previously completed days per user ===
    existing_days = {}
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        for uid, group in df_existing.groupby("user_id"):
            existing_days[uid] = group["day"].max()
    else:
        df_existing = pd.DataFrame()

    for uid in tqdm(range(start_uid, start_uid + num_users), desc=f"Users {start_uid}-{start_uid + num_users - 1}"):
        user_id = f"User{uid:02d}"
        last_completed_day = existing_days.get(user_id, 0)
        if last_completed_day >= num_days:
            print(f"⏩ Skipping {user_id} (all {num_days} days already done)")
            continue

        api_calls = 0
        progression_type = random.choice([
            "StableHealthy", "MildProgressor", "FastDecliner",
            "GradualDecliner", "StableMCI", "StableEarlyAD"
        ])
        print(f"[{user_id}] Progression = {progression_type} (resuming from Day {last_completed_day + 1})")

        for day in range(last_completed_day + 1, num_days + 1):
            date = (datetime(2025, 1, 1) + timedelta(days=day - 1)).strftime("%Y-%m-%d")
            cat1, cat2 = random.sample(categories, 2)
            video_pool = list(set(video_library[cat1] + video_library[cat2]))
            video_titles = random.sample(video_pool, min(5, len(video_pool)))

            daily_data = []
            for video_title in video_titles:
                label = determine_label_for_day(progression_type, day)

                if USE_GROQ_API:
                    api_calls += 1
                    if api_calls % 20 == 0:
                        print(f"⏳ Mid-user cooldown after {api_calls} Groq calls...")
                        time.sleep(random.uniform(60, 120))

                summary = generate_realistic_summary(video_title, label, day)
                coherence = compute_coherence(summary, baseline_vectors[video_title])

                if label == "Healthy":
                    wt, skips, pauses, replays = random.randint(45, 60), random.randint(0, 5), random.randint(0, 2), random.randint(0, 1)
                    liked, shared = int(random.random() < 0.7), int(random.random() < 0.4)
                elif label == "MCI":
                    wt, skips, pauses, replays = random.randint(30, 50), random.randint(5, 15), random.randint(1, 3), random.randint(1, 2)
                    liked, shared = int(random.random() < 0.5), int(random.random() < 0.3)
                else:
                    wt, skips, pauses, replays = random.randint(15, 35), random.randint(10, 25), random.randint(2, 5), random.randint(2, 4)
                    liked, shared = int(random.random() < 0.2), int(random.random() < 0.1)

                daily_data.append({
                    "user_id": user_id,
                    "date": date,
                    "day": day,
                    "category_1": cat1,
                    "category_2": cat2,
                    "video_title": video_title,
                    "summary": summary,
                    "coherence_score": round(coherence, 3),
                    "watch_time_secs": wt,
                    "skipped_secs": skips,
                    "pause_count": pauses,
                    "replay_count": replays,
                    "liked": liked,
                    "shared": shared,
                    "label": label
                })

            # Save after each day to avoid loss
            df_day = pd.DataFrame(daily_data)
            write_mode = 'a' if os.path.exists(path) else 'w'
            header = not os.path.exists(path) or (df_existing.empty and day == 1 and uid == start_uid)
            df_day.to_csv(path, mode=write_mode, index=False, header=header)

            time.sleep(random.uniform(0.8, 2.0))  # per-day delay


# def simulate_users(start_uid, num_users=1, num_days=100, baseline_vectors=None):
#     global USE_GROQ_API
#     all_data = []
#     path = "Lsimulated_AD_users_groq.csv"

#     baseline_summaries = get_or_generate_baseline_summaries()
#     if baseline_vectors is None:
#         baseline_vectors = {
#             v: sbert_model.encode([s])[0] for v, s in baseline_summaries.items()
#         }

#     for uid in tqdm(range(start_uid, start_uid + num_users), desc=f"Users {start_uid}-{start_uid + num_users - 1}"):
#         user_id = f"User{uid:02d}"
#         api_calls = 0  # Track number of Groq requests per user

#         progression_type = random.choice([
#             "StableHealthy", "MildProgressor", "FastDecliner",
#             "GradualDecliner", "StableMCI", "StableEarlyAD"
#         ])
#         print(f"[{user_id}] Progression = {progression_type}")

#         for day in range(1, num_days + 1):
#             date = (datetime(2025, 1, 1) + timedelta(days=day - 1)).strftime("%Y-%m-%d")
#             cat1, cat2 = random.sample(categories, 2)
#             video_pool = list(set(video_library[cat1] + video_library[cat2]))
#             video_titles = random.sample(video_pool, min(5, len(video_pool)))

#             for video_title in video_titles:
#                 label = determine_label_for_day(progression_type, day)

#                 if USE_GROQ_API:
#                     api_calls += 1
#                     if api_calls % 20 == 0:
#                         print(f"⏳ Mid-user cooldown after {api_calls} Groq calls...")
#                         time.sleep(random.uniform(60, 120))  # pause

#                 summary = generate_realistic_summary(video_title, label, day)
#                 coherence = compute_coherence(summary, baseline_vectors[video_title])

#                 if label == "Healthy":
#                     wt, skips, pauses, replays = random.randint(45, 60), random.randint(0, 5), random.randint(0, 2), random.randint(0, 1)
#                     liked, shared = int(random.random() < 0.7), int(random.random() < 0.4)
#                 elif label == "MCI":
#                     wt, skips, pauses, replays = random.randint(30, 50), random.randint(5, 15), random.randint(1, 3), random.randint(1, 2)
#                     liked, shared = int(random.random() < 0.5), int(random.random() < 0.3)
#                 else:
#                     wt, skips, pauses, replays = random.randint(15, 35), random.randint(10, 25), random.randint(2, 5), random.randint(2, 4)
#                     liked, shared = int(random.random() < 0.2), int(random.random() < 0.1)

#                 all_data.append({
#                     "user_id": user_id,
#                     "date": date,
#                     "day": day,
#                     "category_1": cat1,
#                     "category_2": cat2,
#                     "video_title": video_title,
#                     "summary": summary,
#                     "coherence_score": round(coherence, 3),
#                     "watch_time_secs": wt,
#                     "skipped_secs": skips,
#                     "pause_count": pauses,
#                     "replay_count": replays,
#                     "liked": liked,
#                     "shared": shared,
#                     "label": label
#                 })

#             time.sleep(random.uniform(0.8, 2.0))  # per-day delay

#     df = pd.DataFrame(all_data)
#     if os.path.exists(path):
#         df_existing = pd.read_csv(path)
#         df = pd.concat([df_existing, df], ignore_index=True)
#     df.to_csv(path, index=False)
#     print(f"✅ Saved to {path}")

# === Main execution ===
# if __name__ == "__main__":
    # print("🧠 Resuming simulation for users 68–100 over 100 days...")

    # already_done = set()
    # path = "Lsimulated_AD_users_groq.csv"
    # if os.path.exists(path):
    #     df = pd.read_csv(path)
    #     already_done = set(int(uid.replace("User", "")) for uid in df["user_id"].unique())

    # groq_users_all = {1, 4, 7, 12, 15, 19, 22, 23, 31, 39, 41, 65, 68, 72, 75, 78, 84, 87, 90, 94, 97, 100}
    # print(f"✅ GROQ users: {sorted(groq_users_all)}")

    # base_summaries = get_or_generate_baseline_summaries()
    # baseline_vectors = {v: sbert_model.encode([s])[0] for v, s in base_summaries.items()}

    # for i in range(68, 101):
    #     if i in already_done:
    #         print(f"⏩ Skipping User{i:02d} (already simulated)")
    #         continue

    #     USE_GROQ_API = i in groq_users_all
    #     print(f"\n🚦 Simulating User{i:02d} with {'Groq' if USE_GROQ_API else 'Fallback'}...")

    #     try:
    #         simulate_users(start_uid=i, num_users=1, num_days=100, baseline_vectors=baseline_vectors)
    #     except Exception as e:
    #         print(f"❌ Error for User{i:02d}: {e}")
    #         continue

    #     if USE_GROQ_API:
    #         print("🧊 Cooling down after Groq user...")
    #         time.sleep(random.uniform(120, 200))
    #     else:
    #         time.sleep(random.uniform(10, 20))

if __name__ == "__main__":
    print("🧠 Resuming simulation for users 68–100 over 100 days...")

    path = "Lsimulated_AD_users_groq.csv"

    # Determine all already simulated users from file
    already_done_users = set()
    last_days = {}
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        last_days = df_existing.groupby("user_id")["day"].max().to_dict()
        already_done_users = {
            int(uid.replace("User", ""))
            for uid, last_day in last_days.items()
            if last_day >= 100
        }

    # Set of users intended to use Groq
    groq_users_all = {1, 4, 7, 12, 15, 19, 22, 23, 31, 39, 41, 65, 68, 72, 75, 78, 84, 87, 90, 94, 97, 100}
    print(f"✅ GROQ users: {sorted(groq_users_all)}")

    # Load sentence embeddings for baseline
    base_summaries = get_or_generate_baseline_summaries()
    baseline_vectors = {v: sbert_model.encode([s])[0] for v, s in base_summaries.items()}

    for i in range(68, 101):
        if i in already_done_users:
            print(f"⏩ Skipping User{i:02d} (already simulated 100 days)")
            continue

        USE_GROQ_API = i in groq_users_all
        print(f"\n🚦 Simulating User{i:02d} with {'Groq' if USE_GROQ_API else 'Fallback'}...")

        try:
            simulate_users(start_uid=i, num_users=1, num_days=100, baseline_vectors=baseline_vectors)
        except Exception as e:
            print(f"❌ Error for User{i:02d}: {e}")
            continue

        if USE_GROQ_API:
            print("🧊 Cooling down after Groq user...")
            time.sleep(random.uniform(120, 200))
        else:
            time.sleep(random.uniform(10, 20))

