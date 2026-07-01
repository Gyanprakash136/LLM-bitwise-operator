import argparse
import os
import time
import numpy as np
import pickle
import csv
import json

from scoring import calculate_heuristic_score, ensemble_score

try:
    import faiss
except ImportError:
    print("Warning: faiss-cpu not installed.")

try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python not installed. Will fallback to heuristic scoring.")
    Llama = None

def get_candidate_details(candidates_file, target_ids):
    import gzip
    open_func = gzip.open if candidates_file.endswith('.gz') else open
    
    found = {}
    target_set = set(target_ids)
    
    with open_func(candidates_file, 'rt') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            if cid in target_set:
                found[cid] = cand
                if len(found) == len(target_set):
                    break
    return found

def main(candidates_path, output_csv, artifacts_dir="India_runs_data_and_ai_challenge/artifacts"):
    start_time = time.time()
    
    # 1. Load Artifacts
    faiss_index_path = os.path.join(artifacts_dir, "faiss_index.bin")
    mapping_path = os.path.join(artifacts_dir, "faiss_mapping.pkl")
    features_path = os.path.join(artifacts_dir, "candidate_features.pkl")
    jd_query_path = os.path.join(artifacts_dir, "jd_query_vector.npy")
    model_path = os.path.join(artifacts_dir, "finetuned_ranker.Q4_K_M.gguf")
    
    try:
        index = faiss.read_index(faiss_index_path)
        with open(mapping_path, "rb") as f:
            faiss_mapping = pickle.load(f)
        with open(features_path, "rb") as f:
            features_dict = pickle.load(f)
        jd_query = np.load(jd_query_path)
    except Exception as e:
        print(f"Error loading precomputed artifacts: {e}")
        return
        
    # Stage 1: FAISS Retrieval (Top 2000)
    print("Stage 1: FAISS Retrieval...")
    k = 2000
    distances, indices = index.search(jd_query, k)
    
    retrieved_cids = []
    semantic_sims = {}
    
    # Apply hard filter (years_of_experience) on retrieved subset
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        cid = faiss_mapping[idx]
        feat = features_dict.get(cid, {})
        # Exp filter [5,9]
        if 5 <= feat.get("years_of_experience", 0) <= 9:
            retrieved_cids.append(cid)
            semantic_sims[cid] = float(distances[0][i])
            
    # We should have ~1200 left. If too few, we might relax the filter.
    print(f"Candidates after hard filter: {len(retrieved_cids)}")
    
    # Stage 2: Heuristic Pre-score
    print("Stage 2: Heuristic Pre-score...")
    scored_candidates = []
    for cid in retrieved_cids:
        h_score = calculate_heuristic_score(features_dict[cid], semantic_sims[cid])
        scored_candidates.append((cid, h_score))
        
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_200 = scored_candidates[:200]
    
    print("Stage 3: LLM Re-ranking...")
    # Load full candidate data for the top 200 for LLM inference
    top_200_cids = [c[0] for c in top_200]
    candidate_details = get_candidate_details(candidates_path, top_200_cids)
    
    llm = None
    if Llama is not None and os.path.exists(model_path):
        print("Loading local LLM via llama.cpp...")
        llm = Llama(model_path=model_path, n_ctx=512, n_threads=8, verbose=False)
        
    final_results = []
    
    for rank_idx, (cid, h_score) in enumerate(top_200):
        c_detail = candidate_details.get(cid, {})
        profile = c_detail.get("profile", {})
        
        title = profile.get("current_title", "Unknown")
        years = profile.get("years_of_experience", 0)
        skills_list = c_detail.get("skills", [])
        skills = " ".join([s["name"] for s in skills_list[:5]])
        signals = c_detail.get("redrob_signals", {})
        response_rate = signals.get("recruiter_response_rate", 0.0)
        
        fallback_reasoning = f"{title} with {years} yrs; {len(skills_list)} AI core skills; response rate {response_rate:.2f}."

        if llm:
            # Construct prompt for LLM
            prompt = (
                "Given this job description and candidate profile, score fit 0-100 and explain in 1-2 sentences.\n"
                "JD: Senior ML Engineer with production embedding experience.\n"
                f"Candidate: {profile.get('anonymized_name', '')}, {title}, {years} yrs, Skills: [{skills}], Response Rate: {response_rate * 100:.1f}%\nOutput:\n"
            )
            
            output = llm(prompt, max_tokens=60, stop=["\\n", "</s>"])
            text = output["choices"][0]["text"].strip()
            
            # Parse output: "85 | reasoning..."
            try:
                parts = text.split("|", 1)
                llm_score = float(parts[0].strip())
                reasoning = parts[1].strip() if len(parts) > 1 else fallback_reasoning
            except Exception:
                # Parsing failed, fallback
                llm_score = h_score * 100 # scale h_score to 0-100 roughly
                reasoning = fallback_reasoning
        else:
            # Fallback if no LLM
            llm_score = h_score * 100 # assuming h_score is ~0.0-1.0
            reasoning = fallback_reasoning
            
        f_score = ensemble_score(llm_score, h_score * 100)
        final_results.append({
            "candidate_id": cid,
            "score": f_score,
            "reasoning": reasoning
        })
        
    # Sort final results
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Output top 100
    print("Stage 4: Output writing...")
    top_100 = final_results[:100]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, res in enumerate(top_100):
            writer.writerow([res["candidate_id"], i + 1, round(res["score"], 4), res["reasoning"]])
            
    elapsed = time.time() - start_time
    print(f"Ranking complete in {elapsed:.2f} seconds. Output saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--out", default="India_runs_data_and_ai_challenge/submission.csv", help="Output CSV file")
    args = parser.parse_args()
    main(args.candidates, args.out)
