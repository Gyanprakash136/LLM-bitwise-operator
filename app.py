import gradio as gr
import json
import csv
import io
import os
import pandas as pd
from honeypot_detector import is_honeypot

SAMPLE_PATH = "India_runs_data_and_ai_challenge/sample_candidates.json"

# JD key signals for keyword matching
JD_KEYWORDS = {
    "python", "faiss", "embeddings", "llm", "vector", "machine learning",
    "deep learning", "nlp", "transformers", "rag", "production", "ml engineer",
    "senior", "pytorch", "tensorflow", "bert", "openai", "langchain",
    "sentence transformers", "information retrieval", "fine-tuning"
}

def score_candidate(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills_list = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    years = profile.get("years_of_experience", 0)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    github = signals.get("github_activity_score", -1)
    interview_rate = signals.get("interview_completion_rate", 0.0)
    open_to_work = signals.get("open_to_work_flag", False)
    profile_completeness = signals.get("profile_completeness_score", 0) / 100.0
    saved_by = signals.get("saved_by_recruiters_30d", 0)

    # Experience fit (peaks at 5-9 years)
    if 5 <= years <= 9:
        exp_score = 1.0
    else:
        exp_score = max(0.0, 1.0 - abs(years - 7) / 7.0)

    # Semantic keyword match score
    skill_names = " ".join([s.get("name", "").lower() for s in skills_list])
    title = profile.get("current_title", "").lower()
    summary = profile.get("summary", "").lower()
    combined_text = skill_names + " " + title + " " + summary
    matched = sum(1 for kw in JD_KEYWORDS if kw in combined_text)
    keyword_score = min(1.0, matched / 8.0)

    # Skill quality: higher endorsement duration = less keyword stuffing
    if skills_list:
        avg_duration = sum(s.get("duration_months", 0) for s in skills_list) / len(skills_list)
        skill_quality = min(1.0, avg_duration / 24.0)
    else:
        skill_quality = 0.0

    # GitHub
    github_score = (github / 100.0) if github >= 0 else 0.0

    # Open to work bonus
    availability_bonus = 0.05 if open_to_work else 0.0

    # Weighted ensemble
    score = (
        keyword_score       * 0.35 +
        exp_score           * 0.20 +
        response_rate       * 0.15 +
        github_score        * 0.10 +
        skill_quality       * 0.10 +
        profile_completeness * 0.05 +
        interview_rate      * 0.05 +
        availability_bonus
    )
    return round(score, 4)

def build_reasoning(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills_list = candidate.get("skills", [])
    edu_list = candidate.get("education", [])
    career = candidate.get("career_history", [])
    certs = candidate.get("certifications", [])

    title = profile.get("current_title", "Unknown")
    years = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    location = profile.get("location", "")
    response_rate = signals.get("recruiter_response_rate", 0.0)
    github = signals.get("github_activity_score", -1)
    interview_rate = signals.get("interview_completion_rate", -1)
    notice_period = signals.get("notice_period_days", None)
    open_to_work = signals.get("open_to_work_flag", False)
    willing_to_relocate = signals.get("willing_to_relocate", False)
    work_mode = signals.get("preferred_work_mode", "unspecified")
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", None)
    sal_max = salary.get("max", None)
    profile_completeness = signals.get("profile_completeness_score", 0)
    saved_by = signals.get("saved_by_recruiters_30d", 0)

    top_skills = sorted(skills_list, key=lambda x: x.get("endorsements", 0), reverse=True)[:5]
    skill_str = ", ".join([f"{s['name']} ({s.get('proficiency','?')})" for s in top_skills]) or "none listed"

    edu_str = ""
    if edu_list:
        e = edu_list[0]
        edu_str = f"{e.get('degree','')} in {e.get('field_of_study','')} from {e.get('institution','')} [{e.get('tier','unknown')} institution]. "

    prev_titles = list({r.get("title","") for r in career if not r.get("is_current", False)})[:2]
    prev_str = f"Previously: {', '.join(prev_titles)}. " if prev_titles else ""

    cert_str = ""
    if certs:
        cert_str = f"Certifications: {', '.join([c.get('name','') for c in certs[:2]])}. "

    behavioral = f"Recruiter response rate: {response_rate*100:.0f}%"
    if github >= 0:
        behavioral += f"; GitHub activity: {github:.0f}/100"
    if interview_rate >= 0:
        behavioral += f"; interview completion: {interview_rate*100:.0f}%"

    avail = []
    if open_to_work: avail.append("open to work")
    if notice_period is not None: avail.append(f"notice {notice_period}d")
    if willing_to_relocate: avail.append("willing to relocate")
    avail.append(f"{work_mode} preferred")
    if sal_min and sal_max: avail.append(f"salary {sal_min}-{sal_max} LPA")

    return (
        f"{title} at {company} ({industry}) in {location}, {years} yrs exp. "
        f"Top skills: {skill_str}. "
        f"{edu_str}{prev_str}{cert_str}"
        f"Profile {profile_completeness:.0f}% complete, saved by {saved_by} recruiters. "
        f"Signals — {behavioral}. "
        f"{'; '.join(avail).capitalize()}."
    ).strip()

def run_ranking():
    try:
        if not os.path.exists(SAMPLE_PATH):
            return None, f"Error: {SAMPLE_PATH} not found."

        with open(SAMPLE_PATH, "r") as f:
            raw = f.read().strip()

        if raw.startswith('['):
            candidates = json.loads(raw)
        else:
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]

        results = []
        for cand in candidates:
            if is_honeypot(cand):
                continue
            score = score_candidate(cand)
            reasoning = build_reasoning(cand)
            results.append({
                "candidate_id": cand["candidate_id"],
                "score": score,
                "reasoning": reasoning
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:100]

        # Write CSV to string buffer
        out_path = "/tmp/submission.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for i, r in enumerate(top):
                writer.writerow([r["candidate_id"], i + 1, r["score"], r["reasoning"]])

        df = pd.read_csv(out_path)
        return out_path, df.head(10)

    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="Redrob Ranker - Bitwise Developers") as demo:
    gr.Markdown("# Redrob Ranker — Bitwise Developers")
    gr.Markdown(
        "Hybrid candidate ranking pipeline. "
        "Scores candidates using keyword-semantic match, experience fit, "
        "recruiter behavioral signals, GitHub activity, and availability. "
        "Runs fully offline in under 2 seconds."
    )

    run_btn = gr.Button("Run Ranking Pipeline", variant="primary")

    with gr.Row():
        file_output = gr.File(label="Download Full submission.csv")

    with gr.Row():
        df_output = gr.Dataframe(label="Top 10 Candidates Preview")

    run_btn.click(fn=run_ranking, inputs=[], outputs=[file_output, df_output])

if __name__ == "__main__":
    demo.launch()
