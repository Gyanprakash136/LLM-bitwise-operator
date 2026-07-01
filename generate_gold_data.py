import json
import gzip
import random
import os
import argparse

# This script generates 'gold_ranked_pairs.csv' by ranking a subset of candidates offline.
# Since we don't have a provided gold dataset, we bootstrap our own using the JD.

def evaluate_candidate(jd_text, candidate):
    """
    Dummy evaluation function.
    In reality, you would pass this to Gemini/DeepSeek API here to get a high-quality zero-shot score and reasoning.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    # Very simple heuristic to mock a strong LLM's evaluation
    years_exp = profile.get("years_of_experience", 0)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    github = signals.get("github_activity_score", 0)
    
    # Ideal: 5-9 years, high response, high github
    score = 50
    if 5 <= years_exp <= 9:
        score += 20
    score += (response_rate * 20)
    if github > 0:
        score += (github / 10)
        
    reasoning = f"Evaluated offline: {years_exp} years exp, {response_rate*100:.0f}% response rate."
    return min(99, int(score)), reasoning

def main(candidates_file, output_csv, num_samples=1000):
    print(f"Reading candidates from {candidates_file} to bootstrap gold data...")
    open_func = gzip.open if candidates_file.endswith('.gz') else open
    
    candidates = []
    with open_func(candidates_file, 'rt') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            candidates.append(json.loads(line))
            if len(candidates) >= num_samples * 2: # Read a pool to sample from
                break
                
    # Select random sample to evaluate
    sample = random.sample(candidates, min(num_samples, len(candidates)))
    
    jd_text = "Senior ML Engineer with production embedding experience."
    
    print(f"Evaluating {len(sample)} candidates to generate gold ranked pairs...")
    
    results = []
    for cand in sample:
        score, reasoning = evaluate_candidate(jd_text, cand)
        results.append({
            "candidate_id": cand["candidate_id"],
            "score": score,
            "reasoning": reasoning,
            "raw_candidate": cand
        })
        
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Save to CSV
    import csv
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["winner_id", "loser_id", "winner_score", "loser_score", "winner_reasoning", "loser_reasoning"])
        
        # Generate pairs (top vs bottom)
        for i in range(len(results) // 2):
            winner = results[i]
            loser = results[- (i + 1)] # get from bottom
            
            writer.writerow([
                winner["candidate_id"], loser["candidate_id"], 
                winner["score"], loser["score"],
                winner["reasoning"], loser["reasoning"]
            ])
            
    print(f"Successfully generated {len(results)//2} contrastive pairs into {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl")
    parser.add_argument("--out", default="India_runs_data_and_ai_challenge/gold_ranked_pairs.csv")
    args = parser.parse_args()
    main(args.candidates, args.out)
