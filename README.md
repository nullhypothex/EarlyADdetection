# Early Alzheimer's Detection via Digital Behavior and Language Drift

This repository implements a simulation-based framework for early-stage Alzheimer’s Disease (AD) detection using digital behavior signals and language coherence metrics derived from synthetic social media-style video interactions. Inspired by neuropsychological assessments like MMSE and ADAS-Cog, the system tracks attention, memory, and semantic clarity via daily video engagement, free-form summaries, and question-answering tasks.

---

## Project Overview

- **Users Simulated:** 100
- **Duration:** 100 days
- **Videos per Day:** 5 (from 5 categories: News, Sports, Health, etc.)
- **Key Outputs:** Behavioral logs, linguistic summaries, coherence scores, engagement metrics
- **Main Goal:** Predict cognitive states (Healthy, MCI, EarlyAD) from temporal behavior + summary coherence

---

## Key Components

### Data Simulation (`simulate_data4.py`)
- Assigns cognitive progression type to each user
- Simulates watch time, skip, pause, like/share behavior
- Generates summaries using LLM (Groq API) or template-based fillers
- Computes semantic drift using SBERT embeddings

### Evaluation (`train_model.py`, `analysis_language.py`, `train_lbgm.py`, `train_noisy_model.py`, `abalation.py`)
- Trains classifiers (Logistic Regression, LightGBM)
- Computes metrics (F1, accuracy, confusion matrix)
- Analyzes embedding similarity (BLEU, ROUGE-L, SBERT cosine)

### Streamlit UI (`strm3.py`)
- Allows interactive exploration of video tasks
- Users watch videos, submit summaries and QA
- Coherence score calculated against personalized memory baseline

### Visualization
- Plots semantic drift over time (per user)
- Confusion matrices and classifier results
- Boxplots for coherence drift by cognitive state

---

## Cognitive Modeling

- **Progression Types:** StableHealthy, MildProgressor, GradualDecliner, etc.
- **Summary Drift:** Modeled via filler phrases and semantic degradation
- **QA Tasks:** Derived from MMSE-style questions (recall, sequencing, emotion)

---

## Installation

```bash
git clone https://github.com/nullhypothex/EarlyADdetection.git
cd EarlyADdetection
pip install -r requirements.txt
```
