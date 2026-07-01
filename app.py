import gradio as gr
import json
import csv
import os
import pandas as pd
from honeypot_detector import is_honeypot

# JD key signals for keyword matching against Senior ML Engineer JD
JD_KEYWORDS = {
    "python", "faiss", "embeddings", "llm", "vector", "machine learning",
    "deep learning", "nlp", "transformers", "rag", "production", "ml engineer",
    "senior", "pytorch", "tensorflow", "bert", "openai", "langchain",
    "sentence transformers", "information retrieval", "fine-tuning",
    "vector database", "retrieval", "weaviate", "pinecone", "chroma"
}

# ─────────────────────────────────────────────────────────────
# Stage 1: Feature extraction
# ─────────────────────────────────────────────────────────────
def extract_features(candidate):
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})
    skills   = candidate.get("skills", [])
    career   = candidate.get("career_history", [])

    years              = profile.get("years_of_experience", 0)
    response_rate      = signals.get("recruiter_response_rate", 0.0)
    github             = signals.get("github_activity_score", -1)
    interview_rate     = signals.get("interview_completion_rate", 0.0)
    offer_rate         = signals.get("offer_acceptance_rate", -1)
    open_to_work       = signals.get("open_to_work_flag", False)
    profile_complete   = signals.get("profile_completeness_score", 0) / 100.0
    saved_by           = signals.get("saved_by_recruiters_30d", 0)

    return {
        "years": years,
        "response_rate": response_rate,
        "github": github,
        "interview_rate": interview_rate,
        "offer_rate": offer_rate,
        "open_to_work": open_to_work,
        "profile_complete": profile_complete,
        "saved_by": saved_by,
        "skills": skills,
        "profile": profile,
        "signals": signals,
    }

# ─────────────────────────────────────────────────────────────
# Stage 2: Heuristic scoring
# ─────────────────────────────────────────────────────────────
def score_candidate(candidate, features):
    profile = features["profile"]
    years   = features["years"]

    # Experience fit — peaks at 5–9 years
    exp_score = 1.0 if 5 <= years <= 9 else max(0.0, 1.0 - abs(years - 7) / 7.0)

    # Semantic keyword match against JD
    skill_names = " ".join([s.get("name", "").lower() for s in features["skills"]])
    title       = profile.get("current_title", "").lower()
    summary     = profile.get("summary", "").lower()
    combined    = skill_names + " " + title + " " + summary
    matched     = sum(1 for kw in JD_KEYWORDS if kw in combined)
    keyword_score = min(1.0, matched / 8.0)

    # Skill quality: avg endorsement duration penalises keyword stuffers
    if features["skills"]:
        avg_dur = sum(s.get("duration_months", 0) for s in features["skills"]) / len(features["skills"])
        skill_quality = min(1.0, avg_dur / 24.0)
    else:
        skill_quality = 0.0

    github_score    = (features["github"] / 100.0) if features["github"] >= 0 else 0.0
    avail_bonus     = 0.05 if features["open_to_work"] else 0.0

    score = (
        keyword_score               * 0.35 +
        exp_score                   * 0.20 +
        features["response_rate"]   * 0.15 +
        github_score                * 0.10 +
        skill_quality               * 0.10 +
        features["profile_complete"]* 0.05 +
        features["interview_rate"]  * 0.05 +
        avail_bonus
    )
    return round(min(score, 1.0), 4)

# ─────────────────────────────────────────────────────────────
# Stage 3: Reasoning generation
# ─────────────────────────────────────────────────────────────
def build_reasoning(candidate, features):
    profile  = features["profile"]
    signals  = features["signals"]
    skills   = features["skills"]
    edu_list = candidate.get("education", [])
    career   = candidate.get("career_history", [])
    certs    = candidate.get("certifications", [])

    title    = profile.get("current_title", "Unknown")
    years    = features["years"]
    company  = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    location = profile.get("location", "")
    rr       = features["response_rate"]
    github   = features["github"]
    ir       = features["interview_rate"]
    notice   = signals.get("notice_period_days", None)
    relocate = signals.get("willing_to_relocate", False)
    mode     = signals.get("preferred_work_mode", "unspecified")
    salary   = signals.get("expected_salary_range_inr_lpa", {})
    sal_min  = salary.get("min", None)
    sal_max  = salary.get("max", None)

    top_skills = sorted(skills, key=lambda x: x.get("endorsements", 0), reverse=True)[:5]
    skill_str  = ", ".join([f"{s['name']} ({s.get('proficiency','?')})" for s in top_skills]) or "none listed"

    edu_str = ""
    if edu_list:
        e = edu_list[0]
        edu_str = f"{e.get('degree','')} in {e.get('field_of_study','')} from {e.get('institution','')} [{e.get('tier','unknown')} institution]. "

    prev = list({r.get("title","") for r in career if not r.get("is_current", False)})[:2]
    prev_str = f"Previously: {', '.join(prev)}. " if prev else ""

    cert_str = ""
    if certs:
        cert_str = f"Certifications: {', '.join([c.get('name','') for c in certs[:2]])}. "

    beh = f"Response rate {rr*100:.0f}%"
    if github >= 0:   beh += f"; GitHub {github:.0f}/100"
    if ir >= 0:       beh += f"; interview completion {ir*100:.0f}%"

    avail = []
    if features["open_to_work"]: avail.append("open to work")
    if notice is not None:       avail.append(f"notice {notice}d")
    if relocate:                 avail.append("willing to relocate")
    avail.append(f"{mode} preferred")
    if sal_min and sal_max:      avail.append(f"salary {sal_min}–{sal_max} LPA")

    return (
        f"{title} at {company} ({industry}) in {location}, {years} yrs exp. "
        f"Top skills: {skill_str}. "
        f"{edu_str}{prev_str}{cert_str}"
        f"Profile {features['profile_complete']*100:.0f}% complete, "
        f"saved by {features['saved_by']} recruiters. "
        f"Signals — {beh}. "
        f"{'; '.join(avail).capitalize()}."
    ).strip()

# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def run_pipeline(uploaded_file):
    if uploaded_file is None:
        return None, pd.DataFrame(), "Please upload a JSON or JSONL file."

    try:
        with open(uploaded_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except Exception as e:
        return None, pd.DataFrame(), f"File read error: {e}"

    # Parse JSON array or JSONL
    try:
        if raw.startswith('['):
            candidates = json.loads(raw)
        else:
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except Exception as e:
        return None, pd.DataFrame(), f"JSON parse error: {e}"

    if not candidates:
        return None, pd.DataFrame(), "No candidates found in the uploaded file."

    # Stage 1 + 2: extract features, detect honeypots, score
    scored = []
    honeypots = 0
    for cand in candidates:
        if is_honeypot(cand):
            honeypots += 1
            continue
        feats = extract_features(cand)
        score = score_candidate(cand, feats)
        scored.append((cand, feats, score))

    if not scored:
        return None, pd.DataFrame(), "All candidates were flagged as honeypots."

    # Stage 3: sort and take top 100
    scored.sort(key=lambda x: x[2], reverse=True)
    top_100 = scored[:100]

    # Stage 4: build reasoning and write CSV
    out_path = "/tmp/submission.csv"
    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cand, feats, score) in enumerate(top_100, start=1):
            reasoning = build_reasoning(cand, feats)
            cid = cand["candidate_id"]
            writer.writerow([cid, rank, score, reasoning])
            rows.append({"candidate_id": cid, "rank": rank, "score": score, "reasoning": reasoning[:80] + "..."})

    df_preview = pd.DataFrame(rows[:10])
    status = (
        f"Processed {len(candidates)} candidates. "
        f"Flagged {honeypots} honeypots. "
        f"Ranked {len(scored)} valid candidates. "
        f"Top 100 written to CSV."
    )
    return out_path, df_preview, status

# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Redrob Ranker — Bitwise Developers", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Redrob Candidate Ranking System")
    gr.Markdown(
        "**Team: Bitwise Developers** | India Runs Data and AI Challenge\n\n"
        "Upload a candidate JSON or JSONL file. The pipeline will:\n"
        "1. Filter honeypot candidates\n"
        "2. Extract features and score each candidate using a multi-signal heuristic ensemble\n"
        "3. Output the top 100 ranked candidates with detailed reasoning"
    )

    with gr.Row():
        file_input = gr.File(
            label="Upload Candidate File (JSON array or JSONL)",
            file_types=[".json", ".jsonl"]
        )

    with gr.Row():
        run_btn = gr.Button("Run Ranking Pipeline", variant="primary", scale=2)

    with gr.Row():
        status_box = gr.Textbox(label="Pipeline Status", interactive=False)

    with gr.Row():
        file_output = gr.File(label="Download submission.csv")

    with gr.Row():
        df_output = gr.Dataframe(label="Top 10 Preview")

    run_btn.click(
        fn=run_pipeline,
        inputs=[file_input],
        outputs=[file_output, df_output, status_box]
    )

if __name__ == "__main__":
    demo.launch()
