import gradio as gr
import json
import csv
import os
import pandas as pd
from honeypot_detector import is_honeypot

PRELOADED_SAMPLE = "India_runs_data_and_ai_challenge/sample_candidates.json"

JD_KEYWORDS = {
    "python", "faiss", "embeddings", "llm", "vector", "machine learning",
    "deep learning", "nlp", "transformers", "rag", "production", "ml engineer",
    "senior", "pytorch", "tensorflow", "bert", "openai", "langchain",
    "sentence transformers", "information retrieval", "fine-tuning",
    "vector database", "retrieval", "weaviate", "pinecone", "chroma"
}

def extract_features(candidate):
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})
    skills   = candidate.get("skills", [])
    return {
        "years":            profile.get("years_of_experience", 0),
        "response_rate":    signals.get("recruiter_response_rate", 0.0),
        "github":           signals.get("github_activity_score", -1),
        "interview_rate":   signals.get("interview_completion_rate", 0.0),
        "open_to_work":     signals.get("open_to_work_flag", False),
        "profile_complete": signals.get("profile_completeness_score", 0) / 100.0,
        "saved_by":         signals.get("saved_by_recruiters_30d", 0),
        "skills":           skills,
        "profile":          profile,
        "signals":          signals,
    }

def score_candidate(candidate, f):
    years = f["years"]
    exp_score     = 1.0 if 5 <= years <= 9 else max(0.0, 1.0 - abs(years - 7) / 7.0)
    combined      = " ".join([s.get("name","").lower() for s in f["skills"]]) + " " + f["profile"].get("current_title","").lower() + " " + f["profile"].get("summary","").lower()
    keyword_score = min(1.0, sum(1 for kw in JD_KEYWORDS if kw in combined) / 8.0)
    skill_quality = min(1.0, (sum(s.get("duration_months",0) for s in f["skills"]) / len(f["skills"]) / 24.0)) if f["skills"] else 0.0
    github_score  = (f["github"] / 100.0) if f["github"] >= 0 else 0.0
    avail_bonus   = 0.05 if f["open_to_work"] else 0.0
    return round(min(
        keyword_score * 0.35 + exp_score * 0.20 + f["response_rate"] * 0.15 +
        github_score * 0.10 + skill_quality * 0.10 + f["profile_complete"] * 0.05 +
        f["interview_rate"] * 0.05 + avail_bonus, 1.0), 4)

def build_reasoning(candidate, f):
    p = f["profile"]; s = f["signals"]; skills = f["skills"]
    edu   = candidate.get("education", [])
    certs = candidate.get("certifications", [])
    career= candidate.get("career_history", [])
    top5  = sorted(skills, key=lambda x: x.get("endorsements", 0), reverse=True)[:5]
    skill_str = ", ".join([f"{s['name']} ({s.get('proficiency','?')})" for s in top5]) or "none listed"
    edu_str   = f"{edu[0].get('degree','')} in {edu[0].get('field_of_study','')} from {edu[0].get('institution','')} [{edu[0].get('tier','unknown')} institution]. " if edu else ""
    prev      = list({r.get("title","") for r in career if not r.get("is_current",False)})[:2]
    prev_str  = f"Previously: {', '.join(prev)}. " if prev else ""
    cert_str  = f"Certifications: {', '.join([c.get('name','') for c in certs[:2]])}. " if certs else ""
    salary    = s.get("expected_salary_range_inr_lpa", {})
    avail     = []
    if f["open_to_work"]: avail.append("open to work")
    notice = s.get("notice_period_days")
    if notice is not None: avail.append(f"notice {notice}d")
    if s.get("willing_to_relocate"): avail.append("willing to relocate")
    avail.append(f"{s.get('preferred_work_mode','?')} preferred")
    if salary.get("min") and salary.get("max"): avail.append(f"salary {salary['min']}–{salary['max']} LPA")
    beh = f"Response rate {f['response_rate']*100:.0f}%"
    if f["github"] >= 0: beh += f"; GitHub {f['github']:.0f}/100"
    if f["interview_rate"] >= 0: beh += f"; interview completion {f['interview_rate']*100:.0f}%"
    return (
        f"{p.get('current_title','?')} at {p.get('current_company','')} ({p.get('current_industry','')}) "
        f"in {p.get('location','')}, {f['years']} yrs exp. Top skills: {skill_str}. "
        f"{edu_str}{prev_str}{cert_str}"
        f"Profile {f['profile_complete']*100:.0f}% complete, saved by {f['saved_by']} recruiters. "
        f"Signals — {beh}. {'; '.join(avail).capitalize()}."
    ).strip()

def run_pipeline(candidates):
    if not candidates:
        return None, pd.DataFrame(), "No candidates loaded."
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
        return None, pd.DataFrame(), "All candidates flagged as honeypots."
    scored.sort(key=lambda x: x[2], reverse=True)
    top_100 = scored[:100]
    out_path = "/tmp/submission.csv"
    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cand, feats, score) in enumerate(top_100, start=1):
            reasoning = build_reasoning(cand, feats)
            cid = cand["candidate_id"]
            writer.writerow([cid, rank, score, reasoning])
            rows.append({"candidate_id": cid, "rank": rank, "score": score, "reasoning": reasoning[:100] + "..."})
    status = (
        f"Processed {len(candidates)} candidates — "
        f"{honeypots} honeypots removed, "
        f"{len(scored)} valid candidates ranked. "
        f"Top {min(100, len(top_100))} written to CSV."
    )
    return out_path, pd.DataFrame(rows[:10]), status

def run_preloaded():
    if not os.path.exists(PRELOADED_SAMPLE):
        return None, pd.DataFrame(), f"Pre-loaded sample not found at {PRELOADED_SAMPLE}"
    with open(PRELOADED_SAMPLE, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    candidates = json.loads(raw) if raw.startswith('[') else [json.loads(l) for l in raw.splitlines() if l.strip()]
    return run_pipeline(candidates)

def run_uploaded(uploaded_file):
    if uploaded_file is None:
        return None, pd.DataFrame(), "No file uploaded. Use the pre-loaded sample or upload a small JSON/JSONL file."
    with open(uploaded_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        candidates = json.loads(raw) if raw.startswith('[') else [json.loads(l) for l in raw.splitlines() if l.strip()]
    except Exception as e:
        return None, pd.DataFrame(), f"Parse error: {e}"
    return run_pipeline(candidates)

# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Redrob Ranker — Bitwise Developers", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Redrob Candidate Ranking System")
    gr.Markdown("**Team: Bitwise Developers** | India Runs Data and AI Challenge")
    gr.Markdown(
        "### Pipeline\n"
        "1. Honeypot detection and removal\n"
        "2. Feature extraction (experience, skills, behavioral signals)\n"
        "3. Multi-signal heuristic scoring (keyword match, GitHub, response rate, availability)\n"
        "4. Top 100 selection with detailed reasoning"
    )

    gr.Markdown("---")
    gr.Markdown("### Option 1 — Use Pre-loaded Sample (instant, recommended for demo)")
    sample_btn = gr.Button("Run on Pre-loaded Sample Candidates", variant="primary")

    gr.Markdown("---")
    gr.Markdown(
        "### Option 2 — Upload Your Own File\n"
        "Upload a small JSON array or JSONL file (recommended: under 5MB / ~500 candidates). "
        "Do NOT upload the full 100K dataset here — run that locally using `rank.py`."
    )
    file_input  = gr.File(label="Upload Candidate File (.json or .jsonl)", file_types=[".json", ".jsonl"])
    upload_btn  = gr.Button("Run on Uploaded File", variant="secondary")

    gr.Markdown("---")
    status_box  = gr.Textbox(label="Pipeline Status", interactive=False)
    file_output = gr.File(label="Download submission.csv")
    df_output   = gr.Dataframe(label="Top 10 Candidates Preview")

    sample_btn.click(fn=run_preloaded, inputs=[], outputs=[file_output, df_output, status_box])
    upload_btn.click(fn=run_uploaded, inputs=[file_input], outputs=[file_output, df_output, status_box])

if __name__ == "__main__":
    demo.launch()
