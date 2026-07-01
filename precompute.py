import json
import numpy as np
import pickle
import os
import gzip
from honeypot_detector import is_honeypot

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Warning: sentence_transformers or faiss not installed. Run pip install sentence-transformers faiss-cpu")

# JD Parsing - extracted manually or via LLM based on job_description.docx
JD_PARSED = {
    "hard_filters": {"min_exp": 5, "max_exp": 9, "must_have": ["production embeddings"]},
    "soft_prefs": ["open source contributions", "startup background"],
    "anti_signals": ["pure research", "no deployment experience"],
    "raw_text": "[TITLE] Senior ML Engineer AI Ranking [EXP] 7 years [SKILLS] Python FAISS Embeddings LLM [BIO] We need a Senior ML Engineer with production experience in embeddings-based retrieval systems, vector databases, and evaluation frameworks."
}

def extract_features(candidate):
    """
    Extract structured features from a candidate.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    
    years_exp = profile.get("years_of_experience", 0)
    
    # Skill quality ratio
    if len(skills) > 0:
        total_dur = sum(s.get("duration_months", 0) for s in skills)
        skill_quality_ratio = total_dur / len(skills)
    else:
        skill_quality_ratio = 0
        
    # Last active days (approximation based on today vs last_active_date)
    # For hackathon purposes, assume a relative scale or parse date
    # Let's just use response_rate and other direct signals
    response_rate = signals.get("recruiter_response_rate", 0.0)
    github_score = signals.get("github_activity_score", 0)
    
    return {
        "candidate_id": candidate["candidate_id"],
        "years_of_experience": years_exp,
        "skill_quality_ratio": skill_quality_ratio,
        "recruiter_response_rate": response_rate,
        "github_activity_score": github_score,
        "last_active_days": 30 # placeholder since we don't have current date context
    }

def build_field_weighted_text(candidate):
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "")
    years = profile.get("years_of_experience", 0)
    summary = profile.get("summary", "")[:150]
    
    skills = candidate.get("skills", [])
    top_skills = [s.get("name", "") for s in sorted(skills, key=lambda x: x.get("endorsements", 0), reverse=True)[:5]]
    top5_skills = " ".join(top_skills)
    
    text = f"[TITLE] {title} [EXP] {years} years [SKILLS] {top5_skills} [BIO] {summary}"
    return text

def main(candidates_path="India_runs_data_and_ai_challenge/candidates.jsonl", out_dir="India_runs_data_and_ai_challenge/artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading embedding model BAAI/bge-small-en-v1.5...")
    try:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    except Exception as e:
        print(f"Skipping FAISS build - model load failed: {e}")
        return

    features_list = []
    texts_to_embed = []
    candidate_ids = []
    
    open_func = gzip.open if candidates_path.endswith('.gz') else open
    
    print("Processing candidates...")
    open_func = gzip.open if candidates_path.endswith('.gz') else open
    with open_func(candidates_path, 'rt') as f:
        raw = f.read().strip()
    
    # Support both JSON array and JSONL formats
    if raw.startswith('['):
        candidates_iter = json.loads(raw)
    else:
        candidates_iter = [json.loads(line) for line in raw.splitlines() if line.strip()]

    for candidate in candidates_iter:
            # Hard filter: drop honeypots
            if is_honeypot(candidate):
                continue
                
            features = extract_features(candidate)
            text = build_field_weighted_text(candidate)
            
            features_list.append(features)
            texts_to_embed.append(text)
            candidate_ids.append(candidate["candidate_id"])
            
            if len(features_list) % 10000 == 0:
                print(f"Processed {len(features_list)} valid candidates...")
                
    print(f"Total candidates after filtering: {len(features_list)}")
    
    if len(features_list) == 0:
        print("No valid candidates found. Exiting.")
        return
    
    # Save features
    with open(os.path.join(out_dir, "candidate_features.pkl"), "wb") as f:
        pickle.dump({f["candidate_id"]: f for f in features_list}, f)
        
    # Generate Embeddings
    print("Generating embeddings... (this may take a while)")
    embeddings = model.encode(texts_to_embed, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(os.path.join(out_dir, "candidate_embeddings.npy"), embeddings)
    
    # Build FAISS Index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Cosine similarity since normalized
    index.add(embeddings)
    faiss.write_index(index, os.path.join(out_dir, "faiss_index.bin"))
    
    # Save mapping from FAISS index to candidate ID
    with open(os.path.join(out_dir, "faiss_mapping.pkl"), "wb") as f:
        pickle.dump(candidate_ids, f)
        
    # Precompute JD Query Vector
    print("Precomputing JD query vector...")
    jd_embedding = model.encode([JD_PARSED["raw_text"]], normalize_embeddings=True)
    np.save(os.path.join(out_dir, "jd_query_vector.npy"), jd_embedding)
    
    # Save JD parsed JSON
    with open(os.path.join(out_dir, "jd_parsed.json"), "w") as f:
        json.dump(JD_PARSED, f)

    print(f"Precomputation complete. Artifacts saved in {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--out", default="India_runs_data_and_ai_challenge/artifacts", help="Output directory")
    args = parser.parse_args()
    main(candidates_path=args.candidates, out_dir=args.out)
